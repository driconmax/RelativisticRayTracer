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

static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

static void scroll_callback(GLFWwindow*, double, double yoff) {
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

// SDF: plane through origin with normal n (must be normalized)
__device__ __forceinline__ float sdfPlane(const float3& p, const float3& n) {
    // signed distance to plane: dot(p, n)
    return dot3(p, n);
}

// Combine: union (take nearest)
__device__ __forceinline__ float opUnion(float a, float b) { return fminf(a, b); }

// Scene SDF returns distance + ID (which object won)
struct SdfHit {
    float d;
    int id; // 1=sphere, 2=plane
    float3 normal;
    float sphereCenterDistance;
};

__device__ __forceinline__ SdfHit sceneSDF(const float3& p) {
    // Your 2 objects:
    // - sphere at origin
    // - tilted plane (like your disk plane)
    float3 sphereCenter = make_float3(0.0f, 0.0f, 0.0f);
    float sphereR = 0.35f;
    float3 planeN = normalize3(make_float3(0.0f, 0.4f, 0.0f));

    float3 pFromSphereCenter = sub3(p, sphereCenter);
    float sphereCenterDistance = length3(pFromSphereCenter);
    float ds = sphereCenterDistance - sphereR;
    float dp = sdfPlane(p, planeN);

    SdfHit h;
    if (ds < dp) {
        h.d = ds;
        h.id = 1;
        h.normal = normalize3(pFromSphereCenter); // normal from sphere center to current point
    }
    else {
        h.d = dp;
        h.id = 2;
        h.normal = planeN;
        //h.sphereCenterDistance = -1.0f; // not a sphere hit
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

// Raymarch
__device__ __forceinline__ bool raymarch(const float3& ro, const float3& rd, float& t, SdfHit& hitInfo) {
    const int   MAX_STEPS = 128;
    const float MAX_DIST = 500.0f;
    const float GRAV_DIST = 1.0f;
    const float EPS = 1e-3f;

    t = 0.0f;
    hitInfo.d = 0.0f;
    hitInfo.id = 0;
    hitInfo.normal = make_float3(0.0f, 0.0f, 0.0f);
    hitInfo.sphereCenterDistance = -1.0f;
    float3 l_rd = rd;
    for (int i = 0; i < MAX_STEPS; i++) {
        float3 p = add3(ro, mul3(l_rd, t));
        SdfHit h = sceneSDF(p);
        float gravD = h.sphereCenterDistance - GRAV_DIST;

        // Signed-distance safe hit check (works even if camera starts inside a volume)
        if (fabsf(h.d) < EPS) { hitInfo = h; return true; }

        if (h.sphereCenterDistance < GRAV_DIST) {
            //hitInfo = h;
            //hitInfo.id = 1;
            //hitInfo.d = gravD;
            //hitInfo.normal = normalize3(sub3(p, make_float3(0.0f, 0.0f, 0.0f)));
            //return true;

            l_rd = add3(l_rd, mul3(h.normal, (-1 + (GRAV_DIST / h.sphereCenterDistance)) * 0.1f ));
        }


        // If you zoom too far in you can get tiny steps; clamp helps stability
        //float step = fmaxf(fminf(fabsf(h.d), fabsf(gravD)), 1e-4f);

        float step = fmaxf(h.d, 1e-4f);
        t += step;

        if (t > MAX_DIST) break;
    }
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
    float tanHalfFovY
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
    bool hit = raymarch(ro, rd, tHit, sdfHit);

    float3 col = make_float3(0, 0, 0);
    if (hit) {
        float3 p = add3(ro, mul3(rd, tHit));
        float3 n = sdfHit.normal;

        float3 lightDir = normalize3(make_float3(0.4f, 0.8f, 0.2f));
        float diff = fmaxf(0.0f, dot3(n, lightDir));

        // Simple ambient + a tiny fake spec
        float3 viewDir = mul3(rd, -1.0f);
        float3 halfV = normalize3(add3(lightDir, viewDir));
        float spec = powf(fmaxf(0.0f, dot3(n, halfV)), 64.0f);

        if (sdfHit.id == 1) {
            // Sphere (dark-ish, warm highlight)
            col = make_float3(0.05f, 0.05f, 0.06f);
            col = add3(col, mul3(make_float3(1.0f, 0.6f, 0.2f), diff * 0.30f));
            col = add3(col, mul3(make_float3(1.0f, 0.9f, 0.7f), spec * 0.20f));
        }
        else {
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

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt = (float)(now - lastT);
        lastT = now;

        update_camera(win, dt);

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
            tanHalfFovY
            );
        checkCuda(cudaDeviceSynchronize(), "renderKernel sync");

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
