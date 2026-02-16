// main.cu
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void checkGlShader(GLuint s, const char* name) {
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetShaderInfoLog(s, len, nullptr, log.data());
        printf("Shader compile failed (%s):\n%s\n", name, log.data());
        std::exit(1);
    }
}

static void checkGlProgram(GLuint p) {
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetProgramInfoLog(p, len, nullptr, log.data());
        printf("Program link failed:\n%s\n", log.data());
        std::exit(1);
    }
}

__constant__ char c_debugText[256];

static GLuint makeFullscreenProgram() {
    const char* vsSrc = R"GLSL(
        #version 330 core
        out vec2 vUV;
        void main() {
            vec2 p;
            if (gl_VertexID == 0) p = vec2(-1.0, -1.0);
            if (gl_VertexID == 1) p = vec2( 3.0, -1.0);
            if (gl_VertexID == 2) p = vec2(-1.0,  3.0);
            gl_Position = vec4(p, 0.0, 1.0);
            vUV = 0.5 * (p + vec2(1.0));
        }
    )GLSL";

    const char* fsSrc = R"GLSL(
        #version 330 core
        in vec2 vUV;
        out vec4 outColor;
        uniform sampler2D uTex;
        void main() {
            outColor = texture(uTex, vUV);
        }
    )GLSL";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSrc, nullptr);
    glCompileShader(vs);
    checkGlShader(vs, "VS");

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, nullptr);
    glCompileShader(fs);
    checkGlShader(fs, "FS");

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    checkGlProgram(prog);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// -------------------------
// Camera + Input (host)
// -------------------------
struct Camera {
    float3 pos;
    float yaw;   // radians
    float pitch; // radians
    float fovY;  // radians
};

static Camera g_cam;
static bool g_firstMouse = true;
static double g_lastMx = 0.0, g_lastMy = 0.0;
static bool g_mouseLook = true; // toggle with M if you want
static float g_gravInfluence = 0.10f;

static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

static void scroll_callback(GLFWwindow* win, double, double yoff) {
    bool ctrlDown =
        glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

    if (ctrlDown) {
        // Ctrl + wheel: tune gravity influence in small steps.
        const float influenceStep = 0.01f;
        g_gravInfluence += (float)yoff * influenceStep;
        g_gravInfluence = clampf(g_gravInfluence, 0.0f, 1.0f);
        return;
    }

    // wheel up => zoom in => smaller FOV
    const float step = 0.92f; // ~8% per notch
    if (yoff > 0.0) g_cam.fovY *= step;
    else if (yoff < 0.0) g_cam.fovY /= step;

    g_cam.fovY = clampf(g_cam.fovY, 15.0f * 3.1415926f / 180.0f, 110.0f * 3.1415926f / 180.0f);
}

static void mouse_callback(GLFWwindow* win, double mx, double my) {
    if (!g_mouseLook) return;

    if (g_firstMouse) {
        g_lastMx = mx; g_lastMy = my;
        g_firstMouse = false;
    }

    double dx = mx - g_lastMx;
    double dy = my - g_lastMy;
    g_lastMx = mx; g_lastMy = my;

    const float sens = 0.0025f;
    g_cam.yaw += (float)dx * sens;
    g_cam.pitch -= (float)dy * sens;

    // clamp pitch
    const float lim = 1.55f; // ~89 deg
    g_cam.pitch = clampf(g_cam.pitch, -lim, lim);
}

static float3 h_add3(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static float3 h_sub3(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static float3 h_mul3(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }

static float  h_dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float3 h_cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
static float3 h_normalize3(float3 v) {
    float d = v.x * v.x + v.y * v.y + v.z * v.z;
    float inv = 1.0f / sqrtf(fmaxf(d, 1e-12f));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

static void update_camera(GLFWwindow* win, float dt) {
    // Build camera basis from yaw/pitch
    float cy = cosf(g_cam.yaw), sy = sinf(g_cam.yaw);
    float cp = cosf(g_cam.pitch), sp = sinf(g_cam.pitch);

    float3 forward = h_normalize3(make_float3(cy * cp, sp, sy * cp)); // FPS style
    float3 worldUp = make_float3(0, 1, 0);
    float3 right = h_normalize3(h_cross3(forward, worldUp));
    float3 up = h_cross3(right, forward);

    float speed = 3.0f;
    if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 3.0f;

    float3 delta = make_float3(0, 0, 0);

    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) delta = h_add3(delta, forward);
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) delta = h_sub3(delta, forward);
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) delta = h_add3(delta, right);
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) delta = h_sub3(delta, right);
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) delta = h_add3(delta, worldUp);
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) delta = h_sub3(delta, worldUp);

    float len2 = h_dot3(delta, delta);
    if (len2 > 1e-6f) delta = h_mul3(h_normalize3(delta), speed * dt);

    g_cam.pos = h_add3(g_cam.pos, delta);

    // optional toggle mouse look
    static bool prevM = false;
    bool mNow = glfwGetKey(win, GLFW_KEY_M) == GLFW_PRESS;
    if (mNow && !prevM) {
        g_mouseLook = !g_mouseLook;
        g_firstMouse = true;
        glfwSetInputMode(win, GLFW_CURSOR, g_mouseLook ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
    prevM = mNow;
}



// ---------- SDF / Raymarch helpers ----------

__device__ __forceinline__ float3 make3(float x, float y, float z) { return make_float3(x, y, z); }

__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 normalize3(const float3& v) {
    float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float inv = rsqrtf(fmaxf(len2, 1e-8f));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ __forceinline__ float length3(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// SDF: sphere centered at origin
__device__ __forceinline__ float sdfSphere(const float3& p, float r) {
    return length3(p) - r;
}

// Signed distance to plane: dot(n, p - c)
__device__ __forceinline__ float sdfPlane(const float3& p, const float3& c, const float3& n) {
    return dot3(sub3(p, c), n);
}

// Distance to a finite disk (center c, normal n, radius r)
// This is distance-to-surface (non-negative), suitable for raymarch hit tests.
__device__ __forceinline__ float sdfDisk(const float3& p, const float3& c, const float3& n, float r) {
    float3 q = sub3(p, c);
    float planeDist = dot3(q, n);
    float3 inPlane = sub3(q, mul3(n, planeDist));
    float radial = length3(inPlane) - r;

    if (radial <= 0.0f) {
        return fabsf(planeDist);
    }
    return sqrtf(planeDist * planeDist + radial * radial);
}

// Combine: union (take nearest)
__device__ __forceinline__ float opUnion(float a, float b) { return fminf(a, b); }


// Scene SDF returns distance + ID (which object won)
struct SdfHit {
    float d;
    int id; // 1=sphere, 2=plane, 3=disk
    int steps;
    float3 normal;
    float3 snormal;
    float3 center;
    float sphereCenterDistance;
};

__device__ __forceinline__ SdfHit sceneSDF(const float3& p) {
    const bool ENABLE_PLANE = false;
    const bool ENABLE_DISK = true;
    const bool ENABLE_SPHERE = true;

    float3 sphereCenter = make_float3(0.0f, 5.0f, 0.0f);
    float  sphereR = 0.35f;   // horizon radius (visual)
    float3 floorCenter = make_float3(0.0f, 0.0f, 0.0f);
    float3 floorN = normalize3(make_float3(0.0f, 1.0f, 0.0f));
    float  diskR = 2.0f;

    float3 pFromSphereCenter = sub3(p, sphereCenter);
    float  sphereCenterDistance = length3(pFromSphereCenter);

    float ds = ENABLE_SPHERE ? (sphereCenterDistance - sphereR) : 1e30f;
    float dp = ENABLE_PLANE ? sdfPlane(p, floorCenter, floorN) : 1e30f;
    float dd = ENABLE_DISK ? sdfDisk(p, sphereCenter, floorN, diskR) : 1e30f;

    SdfHit h;
    if (ds <= dp && ds <= dd) {
        h.d = ds; h.id = 1;
        h.normal = normalize3(pFromSphereCenter);
        h.center = sphereCenter;
        h.snormal = normalize3(pFromSphereCenter);
    }
    else if (dp <= dd) {
        h.d = dp; h.id = 2;
        h.normal = floorN;
        h.center = sphereCenter;
        h.snormal = normalize3(pFromSphereCenter);
    }
    else {
        h.d = dd; h.id = 3;
        h.normal = floorN;
        h.center = sphereCenter;
        h.snormal = normalize3(pFromSphereCenter);
    }

    h.sphereCenterDistance = sphereCenterDistance;
    return h;
}


// Normal from gradient
__device__ __forceinline__ float3 estimateNormal(const float3& p) {
    const float e = 1e-3f;
    float dx = sceneSDF(add3(p, make3(e, 0, 0))).d - sceneSDF(add3(p, make3(-e, 0, 0))).d;
    float dy = sceneSDF(add3(p, make3(0, e, 0))).d - sceneSDF(add3(p, make3(0, -e, 0))).d;
    float dz = sceneSDF(add3(p, make3(0, 0, e))).d - sceneSDF(add3(p, make3(0, 0, -e))).d;
    return normalize3(make3(dx, dy, dz));
}

__device__ __forceinline__ float checkerXZ(float x, float z, float scale) {
    int xi = (int)floorf(x * scale);
    int zi = (int)floorf(z * scale);
    return ((xi + zi) & 1) ? 1.0f : 0.0f; // 0 or 1
}

__device__ __forceinline__ unsigned int glyphRow5x7(char ch, int row) {
    switch (ch) {
    case '0': { const unsigned int r[7] = { 14,17,19,21,25,17,14 }; return r[row]; }
    case '1': { const unsigned int r[7] = { 4,12,4,4,4,4,14 }; return r[row]; }
    case '2': { const unsigned int r[7] = { 14,17,1,2,4,8,31 }; return r[row]; }
    case '3': { const unsigned int r[7] = { 14,17,1,6,1,17,14 }; return r[row]; }
    case '4': { const unsigned int r[7] = { 2,6,10,18,31,2,2 }; return r[row]; }
    case '5': { const unsigned int r[7] = { 31,16,30,1,1,17,14 }; return r[row]; }
    case '6': { const unsigned int r[7] = { 6,8,16,30,17,17,14 }; return r[row]; }
    case '7': { const unsigned int r[7] = { 31,1,2,4,8,8,8 }; return r[row]; }
    case '8': { const unsigned int r[7] = { 14,17,17,14,17,17,14 }; return r[row]; }
    case '9': { const unsigned int r[7] = { 14,17,17,15,1,2,12 }; return r[row]; }
    case 'A': { const unsigned int r[7] = { 14,17,17,31,17,17,17 }; return r[row]; }
    case 'F': { const unsigned int r[7] = { 31,16,16,30,16,16,16 }; return r[row]; }
    case 'G': { const unsigned int r[7] = { 14,17,16,23,17,17,14 }; return r[row]; }
    case 'N': { const unsigned int r[7] = { 17,25,21,19,17,17,17 }; return r[row]; }
    case 'O': { const unsigned int r[7] = { 14,17,17,17,17,17,14 }; return r[row]; }
    case 'P': { const unsigned int r[7] = { 30,17,17,30,16,16,16 }; return r[row]; }
    case 'S': { const unsigned int r[7] = { 15,16,16,14,1,1,30 }; return r[row]; }
    case ':': { const unsigned int r[7] = { 0,4,0,0,0,4,0 }; return r[row]; }
    case '.': { const unsigned int r[7] = { 0,0,0,0,0,4,0 }; return r[row]; }
    case '-': { const unsigned int r[7] = { 0,0,0,31,0,0,0 }; return r[row]; }
    case ' ': return 0;
    default:  { const unsigned int r[7] = { 31,1,2,4,8,0,8 }; return r[row]; } // '?'
    }
}

__device__ __forceinline__ bool debugTextPixelOn(int x, int y, int w, int h) {
    const int startX = 10;
    const int startY = 10;
    const int scale = 2;
    const int charW = 5 * scale;
    const int charH = 7 * scale;
    const int advX = 6 * scale;
    const int advY = 8 * scale;

    int cx = startX;
    int cy = startY;
    int yTop = (h - 1) - y; // framebuffer is vertically flipped when presented via GL

    for (int i = 0; i < 256; i++) {
        char ch = c_debugText[i];
        if (ch == '\0') break;

        if (ch == '\n') {
            cx = startX;
            cy += advY;
            if (cy + charH >= h) break;
            continue;
        }

        int lx = x - cx;
        int ly = yTop - cy;
        if (lx >= 0 && lx < charW && ly >= 0 && ly < charH) {
            int gx = lx / scale;
            int gy = ly / scale;
            unsigned int rowBits = glyphRow5x7(ch, gy);
            if (rowBits & (1u << (4 - gx))) return true;
        }

        cx += advX;
        if (cx + charW >= w) {
            cx = startX;
            cy += advY;
            if (cy + charH >= h) break;
        }
    }

    return false;
}



// Perlin Noise

__device__ __forceinline__ float fractf(float x) { return x - floorf(x); }
__device__ __forceinline__ float lerpf(float a, float b, float t) { return a + (b - a) * t; }
__device__ __forceinline__ float smoothf(float t) { return t * t * (3.0f - 2.0f * t); }

__device__ __forceinline__ float hash1(float n) {
    return fractf(sinf(n) * 43758.5453123f);
}

__device__ __forceinline__ float hash3i(int x, int y, int z) {
    // integer hash -> [0,1)
    int n = x * 15731 + y * 789221 + z * 1376312589;
    n = (n << 13) ^ n;
    return 0.5f * (1.0f + (float)((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

__device__ __forceinline__ float valueNoise3(float3 p) {
    // 3D value noise (trilinear + smoothstep)
    int ix = (int)floorf(p.x);
    int iy = (int)floorf(p.y);
    int iz = (int)floorf(p.z);

    float fx = p.x - ix;
    float fy = p.y - iy;
    float fz = p.z - iz;

    float ux = smoothf(fx);
    float uy = smoothf(fy);
    float uz = smoothf(fz);

    float c000 = hash3i(ix, iy, iz);
    float c100 = hash3i(ix + 1, iy, iz);
    float c010 = hash3i(ix, iy + 1, iz);
    float c110 = hash3i(ix + 1, iy + 1, iz);

    float c001 = hash3i(ix, iy, iz + 1);
    float c101 = hash3i(ix + 1, iy, iz + 1);
    float c011 = hash3i(ix, iy + 1, iz + 1);
    float c111 = hash3i(ix + 1, iy + 1, iz + 1);

    float x00 = lerpf(c000, c100, ux);
    float x10 = lerpf(c010, c110, ux);
    float x01 = lerpf(c001, c101, ux);
    float x11 = lerpf(c011, c111, ux);

    float y0 = lerpf(x00, x10, uy);
    float y1 = lerpf(x01, x11, uy);

    return lerpf(y0, y1, uz);
}

__device__ __forceinline__ float fbm3(float3 p) {
    // Fractal Brownian Motion: 5 octaves
    float sum = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    for (int i = 0; i < 5; i++) {
        sum += amp * valueNoise3(mul3(p, freq));
        freq *= 2.02f;
        amp *= 0.5f;
    }
    return sum; // ~[0,1]
}



__device__ __forceinline__ float3 skyColor(float3 rd)
{
    rd = normalize3(rd);

    // --- Base deep space gradient (subtle) ---
    float t = 0.5f * (rd.y + 1.0f);
    float3 col = add3(mul3(make_float3(0.01f, 0.01f, 0.015f), (1.0f - t)),
        mul3(make_float3(0.02f, 0.03f, 0.06f), t));

    // --- Stars (hash in spherical-ish space) ---
    // Map direction to a repeatable 2D coordinate
    float u = 0.5f + atan2f(rd.z, rd.x) * (0.5f / 3.1415926f);
    float v = 0.5f - asinf(rd.y) * (1.0f / 3.1415926f);

    // Scale controls star density
    float starScale = 900.0f;
    float su = u * starScale;
    float sv = v * starScale;

    int ix = (int)floorf(su);
    int iy = (int)floorf(sv);
    float fx = fractf(su);
    float fy = fractf(sv);

    // Random point inside cell
    float sx = hash1((float)(ix * 157 + iy * 113) + 1.17f);
    float sy = hash1((float)(ix * 269 + iy * 353) + 9.73f);

    // Random "does a star exist in this cell?"
    float rnd = hash1((float)(ix * 1973 + iy * 9277) + 0.123f);
    float appear = (rnd > 0.985f) ? 1.0f : 0.0f; // fewer stars = fewer artifacts

    float dx = fx - sx;
    float dy = fy - sy;
    float d2 = dx * dx + dy * dy;

    // Hard radius + soft core
    float r = 0.0018f;                   // star radius in cell space
    float core = expf(-d2 / (r * r));       // gaussian
    float star = appear * core;

    // Optional: clamp so no big shapes ever
    star = fminf(star, 2.0f);

    // A few "big" stars: second layer with lower density
    float bigScale = 120.0f;
    float bu = u * bigScale;
    float bv = v * bigScale;

    int bix = (int)floorf(bu);
    int biy = (int)floorf(bv);
    float bfx = fractf(bu);
    float bfy = fractf(bv);

    float br = hash1((float)(bix * 1597 + biy * 3571) + 4.2f);
    float bsx = hash1((float)(bix * 1013 + biy * 1619) + 2.8f);
    float bsy = hash1((float)(bix * 2029 + biy * 1237) + 6.1f);

    float bdx = bfx - bsx;
    float bdy = bfy - bsy;
    float bd2 = bdx * bdx + bdy * bdy;

    float bAppear = (br > 0.992f) ? 1.0f : 0.0f;
    float bR = 0.02f; // bigger radius (big cells, so bigger looks fine)
    float bigStar = bAppear * expf(-bd2 / (bR * bR));
    col = add3(col, mul3(make_float3(1, 1, 1), bigStar * 0.8f));


    // --- Nebula (fbm in "direction space") ---
    // Use rd as a 3D coordinate; offset & scale to place nebulas
    float3 np = mul3(rd, 6.0f);
    np = add3(np, make_float3(12.3f, -3.7f, 5.1f));

    float n1 = fbm3(np);
    float n2 = fbm3(add3(mul3(np, 1.7f), make_float3(7.0f, 2.0f, -1.0f)));
    float neb = n1 * 0.65f + n2 * 0.35f;

    // Make it patchy: push low values to 0, keep high wisps
    float mask = fmaxf(0.0f, (neb - 0.55f) / (1.0f - 0.55f)); // 0..1
    mask = mask * mask;

    // Color ramp for nebula
    float3 cA = make_float3(0.25f, 0.05f, 0.35f); // purple
    float3 cB = make_float3(0.05f, 0.25f, 0.35f); // teal
    float mixv = fminf(fmaxf((neb - 0.5f) * 2.0f, 0.0f), 1.0f);
    float3 nebCol = add3(mul3(cA, (1.0f - mixv)), mul3(cB, mixv));

    // Add subtle brightness tied to noise
    col = add3(col, mul3(nebCol, mask * 0.35f));

    return col;
}

// Raymarch
__device__ __forceinline__
bool raymarch(const float3& ro, float3& rd0, float& t, SdfHit& hitInfo, float gravInfluence)
{
    const int   MAX_STEPS = 64;      // more steps helps lensing
    const float MAX_DIST = 500.0f;
    const float EPS = 1e-3f;

    // Lens region around your sphere center
    const float3 sphereCenter = make_float3(0.0f, 5.0f, 0.0f);
    const float  GRAV_DIST = 6.0f;   // tighter region helps "fold"
    const float  MIN_STEP = 1e-3f;
    const float  MAX_STEP = 1.0f;  // critical: prevents skipping tight bends

    const float HORIZON_R = 0.35f;
    const float ABSORB_R = HORIZON_R * 1.05f;

    float3 rd = normalize3(rd0);
    float3 p = ro;
    t = 0.0f;

    for (int i = 0; i < MAX_STEPS; i++) {
        SdfHit h = sceneSDF(p);

        if (fabsf(h.d) < EPS) {
            hitInfo = h;
            hitInfo.steps = i;
            return true;
        }

        // ---- lensing / gravity bend ----
        float3 rvec = sub3(p, sphereCenter);
        float  r = length3(rvec);

        if (r < ABSORB_R) {
            hitInfo.id = 99;               // absorbed
            t = 0.0f;                      // not used
            rd0 = rd;
            return true;
        }

        if (r < GRAV_DIST) {
            float rSafe = fmaxf(r, 0.05f);

            // direction toward center
            float3 toCenter = mul3(rvec, -1.0f / rSafe);

            // inverse-square-ish strength (tweakable)
            float strength = gravInfluence / (rSafe * rSafe);
        }

        // SDF step (always positive)
        float step = fmaxf(h.d, MIN_STEP);
        step = fminf(step, MAX_STEP);

        // Apply bend AFTER we know step (so it integrates nicely)
        {
            float3 rvec2 = sub3(p, sphereCenter);
            float  r2 = length3(rvec2);
            if (r2 < GRAV_DIST) {
                float rSafe = fmaxf(r2, 0.05f);
                float3 toCenter = mul3(rvec2, -1.0f / rSafe);
                float strength = gravInfluence / (rSafe * rSafe);

                // integrate
                rd = normalize3(add3(rd, mul3(toCenter, strength * step)));
            }
        }

        // advance
        p = add3(p, mul3(rd, step));
        t += step;

        if (t > MAX_DIST) break;
    }

    rd0 = rd;
    return false;
}


// -------------------------
// CUDA raytracing kernel
// -------------------------
__device__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ float3 f3add(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 f3sub(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 f3mul(float3 a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
__device__ float  f3dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 f3norm(float3 v) {
    float d = v.x * v.x + v.y * v.y + v.z * v.z;
    float inv = rsqrtf(fmaxf(d, 1e-12f));
    return f3(v.x * inv, v.y * inv, v.z * inv);
}
__device__ float3 f3cross(float3 a, float3 b) {
    return f3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ uchar4 packRGB(float3 c) {
    c.x = fminf(fmaxf(c.x, 0.f), 1.f);
    c.y = fminf(fmaxf(c.y, 0.f), 1.f);
    c.z = fminf(fmaxf(c.z, 0.f), 1.f);
    return make_uchar4((unsigned char)(c.x * 255.f), (unsigned char)(c.y * 255.f), (unsigned char)(c.z * 255.f), 255);
}

__device__ bool intersectSphere(float3 ro, float3 rd, float3 center, float radius, float& tHit, float3& nHit) {
    float3 oc = f3sub(ro, center);
    float b = f3dot(oc, rd);
    float c = f3dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.f) return false;
    float s = sqrtf(disc);
    float t0 = -b - s;
    float t1 = -b + s;
    float t = (t0 > 0.001f) ? t0 : ((t1 > 0.001f) ? t1 : -1.f);
    if (t < 0.f) return false;
    tHit = t;
    float3 p = f3add(ro, f3mul(rd, t));
    nHit = f3norm(f3sub(p, center));
    return true;
}

__device__ bool intersectPlane(float3 ro, float3 rd, float3 n, float d, float& tHit) {
    // plane: dot(n, x) + d = 0
    float denom = f3dot(n, rd);
    if (fabsf(denom) < 1e-6f) return false;
    float t = -(f3dot(n, ro) + d) / denom;
    if (t < 0.001f) return false;
    tHit = t;
    return true;
}

__global__ void renderKernel(
    uchar4* out, int w, int h,
    float3 camPos, float3 camFwd, float3 camRight, float3 camUp,
    float tanHalfFovY,
    float gravInfluence
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float u = (x + 0.5f) / (float)w;
    float v = (y + 0.5f) / (float)h;
    //v = 1.0f - v;

    float sx = u * 2.0f - 1.0f;
    float sy = v * 2.0f - 1.0f;

    float aspect = (float)w / (float)h;

    float px = sx * aspect * tanHalfFovY;
    float py = sy * tanHalfFovY;

    float3 ro = camPos;
    float3 rd = normalize3(
        add3(
            add3(camFwd, mul3(camRight, px)),
            mul3(camUp, py)
        )
    );

    float tHit;
    SdfHit sdfHit;
    bool hit = raymarch(ro, rd, tHit, sdfHit, gravInfluence);

    float3 col = make_float3(0, 0, 0);
    if (hit) {
        if (sdfHit.id == 99) {
            col = make_float3(0, 0, 0);
        }
        else {
            // normal shading
            float3 p = add3(ro, mul3(rd, tHit));
            float3 n = sdfHit.normal;

            float3 lightDir = normalize3(make_float3(0.4f, 0.8f, 0.2f));
            float diff = fmaxf(0.0f, dot3(n, lightDir));

            // Simple ambient + a tiny fake spec
            float3 viewDir = mul3(rd, -1.0f);
            float3 halfV = normalize3(add3(lightDir, viewDir));
            float spec = powf(fmaxf(0.0f, dot3(n, halfV)), 64.0f);

            if (sdfHit.id == 1) {
                // Sphere (black)
                col = make_float3(0.0f, 0.0f, 0.0f);
            }
            else if (sdfHit.id == 3) {
                // Disk material (red)
                float3 base = make_float3(0.75f, 0.08f, 0.08f);
                col = add3(mul3(base, 0.20f), mul3(base, diff * 0.95f));
                col = add3(col, mul3(make_float3(1.0f, 0.85f, 0.85f), spec * 0.12f));

                float distFromCenter = length3(sub3(p, sdfHit.center));
                float glow = expf(-distFromCenter * 1.2f);
                col = add3(col, mul3(make_float3(1.0f, 0.4f, 0.1f), glow * 0.5f));

            }
            else {
                // Plane checkerboard using XZ
                // (plane is id == 2)
                // Floor checkerboard using XZ
                float check = checkerXZ(p.x, p.z, 2.0f); // scale: squares per unit
                float3 c0 = make_float3(0.05f, 0.05f, 0.055f); // dark tile
                float3 c1 = make_float3(0.85f, 0.85f, 0.88f);  // light tile
                float3 base = (check > 0.5f) ? c1 : c0;

                // Slight distance fade for nicer look
                float fog = fminf(tHit / 30.0f, 1.0f);
                base = add3(mul3(base, (1.0f - fog)), mul3(make_float3(0.02f, 0.02f, 0.03f), fog));

                col = add3(mul3(base, 0.25f), mul3(base, diff * 0.95f));
                col = add3(col, mul3(make_float3(1.0f, 1.0f, 1.0f), spec * 0.10f));
            }

            // Ambient term
            col = add3(col, make_float3(0.03f, 0.03f, 0.04f));

            //col = make_float3(0, (float)sdfHit.steps / (float)32, 0);
        }
    }
    else {
        col = skyColor(rd);
    }

    if (debugTextPixelOn(x, y, w, h)) {
        col = make_float3(1.0f, 1.0f, 0.25f);
    }

    out[y * w + x] = make_uchar4(
        (unsigned char)(fminf(1.0f, col.x) * 255.0f),
        (unsigned char)(fminf(1.0f, col.y) * 255.0f),
        (unsigned char)(fminf(1.0f, col.z) * 255.0f),
        255
    );
}



int main() {
    const int W = 1280;
    const int H = 720;

    if (!glfwInit()) {
        printf("Failed to init GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(W, H, "CUDA -> OpenGL (3D Renderer)", nullptr, nullptr);
    if (!win) {
        printf("Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("Failed to load GLAD\n");
        return 1;
    }

    glfwSetScrollCallback(win, scroll_callback);
    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Camera init
    g_cam.pos = make_float3(0.0f, 1.2f, 5.0f);
    g_cam.yaw = 3.1415926f; // look toward -Z
    g_cam.pitch = 0.0f;
    g_cam.fovY = 60.0f * 3.1415926f / 180.0f;

    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    // GL texture
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    cudaGraphicsResource* cudaTexRes = nullptr;
    checkCuda(cudaGraphicsGLRegisterImage(&cudaTexRes, tex, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterImage");

    GLuint prog = makeFullscreenProgram();
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);

    // Device buffer
    uchar4* d_pixels = nullptr;
    size_t pixelBytes = (size_t)W * (size_t)H * sizeof(uchar4);
    checkCuda(cudaMalloc(&d_pixels, pixelBytes), "cudaMalloc d_pixels");

    double lastT = glfwGetTime();
    double dbgAccum = 0.0;
    int dbgFrames = 0;
    float dbgFps = 0.0f;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt = (float)(now - lastT);
        lastT = now;
        dbgAccum += dt;
        dbgFrames++;

        update_camera(win, dt);

        // Debug text in window title: FPS + camera transform.
        if (dbgAccum >= 0.20) {
            dbgFps = (float)dbgFrames / (float)dbgAccum;
            float yawDeg = g_cam.yaw * (180.0f / 3.1415926f);
            float pitchDeg = g_cam.pitch * (180.0f / 3.1415926f);
            char title[256];
            snprintf(
                title, sizeof(title),
                "CUDA -> OpenGL (3D Renderer) | FPS: %.1f | Pos: (%.2f, %.2f, %.2f) | Yaw: %.1f deg | Pitch: %.1f deg",
                dbgFps, g_cam.pos.x, g_cam.pos.y, g_cam.pos.z, yawDeg, pitchDeg
            );
            glfwSetWindowTitle(win, title);
            dbgAccum = 0.0;
            dbgFrames = 0;
        }

        {
            float yawDeg = g_cam.yaw * (180.0f / 3.1415926f);
            float pitchDeg = g_cam.pitch * (180.0f / 3.1415926f);
            char debugTextHost[256];
            snprintf(
                debugTextHost, sizeof(debugTextHost),
                "FPS: %.1f\nPOS: %.2f %.2f %.2f\nANG: %.1f %.1f",
                dbgFps, g_cam.pos.x, g_cam.pos.y, g_cam.pos.z, yawDeg, pitchDeg
            );
            checkCuda(
                cudaMemcpyToSymbol(c_debugText, debugTextHost, sizeof(debugTextHost), 0, cudaMemcpyHostToDevice),
                "cudaMemcpyToSymbol c_debugText"
            );
        }

        // Camera basis
        float cy = cosf(g_cam.yaw), sy = sinf(g_cam.yaw);
        float cp = cosf(g_cam.pitch), sp = sinf(g_cam.pitch);

        float3 fwd = h_normalize3(make_float3(cy * cp, sp, sy * cp));
        float3 worldUp = make_float3(0, 1, 0);
        float3 right = h_normalize3(h_cross3(fwd, worldUp));
        float3 up = h_cross3(right, fwd);

        float tanHalfFovY = tanf(g_cam.fovY * 0.5f);

        // CUDA render
        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

        renderKernel << <grid, block >> > (
            d_pixels, W, H,
            g_cam.pos, fwd, right, up,
            tanHalfFovY,
            g_gravInfluence
            );
        //checkCuda(cudaDeviceSynchronize(), "renderKernel sync");
        checkCuda(cudaGetLastError(), "renderKernel launch");


        // Copy to GL texture
        checkCuda(cudaGraphicsMapResources(1, &cudaTexRes, 0), "cudaGraphicsMapResources");

        cudaArray_t arr = nullptr;
        checkCuda(cudaGraphicsSubResourceGetMappedArray(&arr, cudaTexRes, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray");

        checkCuda(cudaMemcpy2DToArray(arr, 0, 0,
            d_pixels, W * sizeof(uchar4),
            W * sizeof(uchar4), H,
            cudaMemcpyDeviceToDevice),
            "cudaMemcpy2DToArray");

        checkCuda(cudaGraphicsUnmapResources(1, &cudaTexRes, 0), "cudaGraphicsUnmapResources");

        // Draw fullscreen
        glViewport(0, 0, W, H);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(win);
    }

    cudaFree(d_pixels);
    cudaGraphicsUnregisterResource(cudaTexRes);

    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
