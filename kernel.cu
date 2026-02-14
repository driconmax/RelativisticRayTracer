// main.cu
#include <cstdio>
#include <vector>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

static float g_zoom = 1.0f; // 1 = default, >1 zoom in, <1 zoom out

static void scroll_callback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset)
{
    // yoffset: + = wheel up, - = wheel down
    // Exponential feels nice and avoids negative/zero zoom.
    const float step = 1.10f; // 10% per notch
    if (yoffset > 0.0) g_zoom *= step;
    else if (yoffset < 0.0) g_zoom /= step;

    // Clamp so it never explodes / becomes microscopic
    if (g_zoom < 0.05f) g_zoom = 0.05f;
    if (g_zoom > 500.0f) g_zoom = 500.0f;
}

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
            // Fullscreen triangle trick (no VBO needed)
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

__device__ __forceinline__ float safe_r(float2 p) {
    float r2 = p.x * p.x + p.y * p.y;
    return sqrtf(fmaxf(r2, 1e-8f));
}

__device__ __forceinline__ float safe_div(float a, float b) {
    return a / (fabsf(b) < 1e-6f ? (b < 0 ? -1e-6f : 1e-6f) : b);
}

__device__ __forceinline__ bool finite2(float2 v) {
    return isfinite(v.x) && isfinite(v.y);
}

__device__ float length2(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ __forceinline__ float2 normalize2(float2 v) {
    float r2 = v.x * v.x + v.y * v.y;
    float inv = rsqrtf(fmaxf(r2, 1e-8f));
    return make_float2(v.x * inv, v.y * inv);
}


__global__ void renderKernel(uchar4* out, int w, int h, float time, float2 bhCenter, float zoom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float u = (x + 0.5f) / w;
    float v = (y + 0.5f) / h;

    // Convert to centered coordinates, then apply zoom (zoom in = smaller world units)
    float2 s;
    s.x = (u - 0.5f) * 2.0f;
    s.y = (v - 0.5f) * 2.0f * (float)h / (float)w;

    // screen-space pan (mouse)
    s.x -= bhCenter.x;
    s.y -= bhCenter.y;

    // FOV zoom: bigger zoom => narrower FOV => “closer”
    s.x /= zoom;
    s.y /= zoom;

    // now treat this as your “p”
    float2 p = s;



    float r = safe_r(p);


    const float M = 0.2f; // black hole mass
    const float horizon = 2.0f * M;

    // If inside event horizon → black
    if (r < horizon) {
        out[y * w + x] = make_uchar4(0, 0, 0, 255);
        return;
    }

    float bendStrength = (M * 2.5f) / (r * r + 0.02f);
    float2 bent = make_float2(
        p.x - bendStrength * safe_div(p.x, r),
        p.y - bendStrength * safe_div(p.y, r)
    );

    if (!finite2(bent)) {
        out[y * w + x] = make_uchar4(0, 0, 0, 255);
        return;
    }



    float photonSphere = horizon * 1.5f;
    float photonBoost = 0.0f;

    float d = fabsf(r - photonSphere);
    if (d < 0.05f)
    {
        photonBoost = 1.0f - d / 0.05f;
    }


    // ---- Physical-ish disk (robust) ----

    // Camera at z=+3 looking toward -z
    float3 rayOrigin = make_float3(0.0f, 0.0f, 3.0f);

    // Build a forward ray. Ensure z is negative.
    float3 rayDir = make_float3(p.x, p.y, -1.0f);
    rayDir.x = 0.6f * rayDir.x + 0.4f * bent.x;
    rayDir.y = 0.6f * rayDir.y + 0.4f * bent.y;


    // Normalize
    float invLen = rsqrtf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z);
    rayDir.x *= invLen;
    rayDir.y *= invLen;
    rayDir.z *= invLen;

    // Tilted disk normal
    float3 diskNormal = make_float3(0.0f, 0.4f, 1.0f);

    // normalize
    float invN = rsqrtf(diskNormal.x * diskNormal.x + diskNormal.y * diskNormal.y + diskNormal.z * diskNormal.z);
    diskNormal.x *= invN;
    diskNormal.y *= invN;
    diskNormal.z *= invN;

    // Plane equation: dot(n, X) = 0  (plane through origin)

    float denom = rayDir.x * diskNormal.x +
        rayDir.y * diskNormal.y +
        rayDir.z * diskNormal.z;

    float disk = 0.0f;

    if (fabsf(denom) > 1e-4f) {
        float tHit = -(
            rayOrigin.x * diskNormal.x +
            rayOrigin.y * diskNormal.y +
            rayOrigin.z * diskNormal.z
            ) / denom;

        if (tHit > 0.0f) {
            float hitx = rayOrigin.x + rayDir.x * tHit;
            float hity = rayOrigin.y + rayDir.y * tHit;
            float hitz = rayOrigin.z + rayDir.z * tHit;

            float diskR = sqrtf(hitx * hitx + hity * hity + hitz * hitz);

            diskR *= 0.25f;

            float diskInner = horizon * 1.2f;
            float diskOuter = horizon * 8.0f;

            if (diskR > diskInner && diskR < diskOuter) {
                float x01 = (diskR - diskInner) / (diskOuter - diskInner);
                float falloff = 1.0f - x01;
                disk = powf(falloff, 2.0f);
            }
        }
    }




    //Starmap
    float star = 0.0f;

    float2 sky = normalize2(bent);              // or normalize2(make_float2(rayDir.x, rayDir.y));
    float sx = sky.x * 50.0f;
    float sy = sky.y * 50.0f;


    float noise = sinf(sx * 12.9898f + sy * 78.233f) * 43758.5453f;
    noise = noise - floorf(noise);

    if (noise > 0.995f)
        star = 1.0f;

    unsigned char R = (unsigned char)((star) * 255);
    unsigned char G = (unsigned char)((star) * 255);
    unsigned char B = (unsigned char)((star) * 255);
    
    // Add disk glow (warm)
    int rI = (int)R + (int)(disk * 255.0f);
    int gI = (int)G + (int)(disk * 140.0f);
    int bI = (int)B + (int)(disk * 40.0f);
    float doppler = rayDir.x * 0.5f + 0.5f; // left-right asymmetry
    float redShift = 0.5f + 0.5f * doppler;

    rI += disk * 255.0f * redShift;
    gI += disk * 120.0f * redShift;
    bI += disk * 30.0f * redShift;


    R = (unsigned char)min(255, rI);
    G = (unsigned char)min(255, gI);
    B = (unsigned char)min(255, bI);
    
    int pr = (int)R + (int)(photonBoost * 255.0f);
    int pg = (int)G + (int)(photonBoost * 200.0f);
    int pb = (int)B + (int)(photonBoost * 120.0f);

    R = min(255, pr);
    G = min(255, pg);
    B = min(255, pb);



    out[y * w + x] = make_uchar4(R, G, B, 255);
}

int main() {
    const int W = 1280;
    const int H = 720;

    // --- GLFW / OpenGL init ---
    if (!glfwInit()) {
        printf("Failed to init GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(W, H, "CUDA -> OpenGL (Realtime)", nullptr, nullptr);
    if (!win) {
        printf("Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSetScrollCallback(win, scroll_callback);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("Failed to load GLAD\n");
        return 1;
    }

    // --- CUDA init (must be after GL context for interop) ---
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    // --- Create GL texture ---
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Register texture with CUDA
    cudaGraphicsResource* cudaTexRes = nullptr;
    checkCuda(cudaGraphicsGLRegisterImage(&cudaTexRes, tex, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterImage");

    // Create shader program and a dummy VAO (required in core profile)
    GLuint prog = makeFullscreenProgram();
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);

    // Device buffer we render into (uchar4 per pixel)
    uchar4* d_pixels = nullptr;
    size_t pixelBytes = (size_t)W * (size_t)H * sizeof(uchar4);
    checkCuda(cudaMalloc(&d_pixels, pixelBytes), "cudaMalloc d_pixels");

    float t = 0.0f;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        double mx, my;
        glfwGetCursorPos(win, &mx, &my);

        // mx,my are in pixels (0..W, 0..H), with y downward.
        // Convert to the same coordinate space as p BEFORE the "- 0.5" scaling:
        // p.x uses (u-0.5)*2  => range [-1..1]
        // p.y uses (v-0.5)*2*(H/W) => range [-H/W .. H/W]
        float cx = (float)((mx / (double)W) * 2.0 - 1.0);
        float cy = (float)(((1.0 - my / (double)H) * 2.0 - 1.0) * ((double)H / (double)W));
        float2 bhCenter = make_float2(cx, cy);




        // CUDA render
        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        renderKernel <<<grid, block >>> (d_pixels, W, H, t, bhCenter, g_zoom);
        checkCuda(cudaDeviceSynchronize(), "renderKernel sync");

        // Map the GL texture for CUDA and copy GPU->GPU
        checkCuda(cudaGraphicsMapResources(1, &cudaTexRes, 0), "cudaGraphicsMapResources");

        cudaArray_t arr = nullptr;
        checkCuda(cudaGraphicsSubResourceGetMappedArray(&arr, cudaTexRes, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray");

        // Copy linear device buffer into the CUDA array backing the GL texture
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

        t += 0.016f; // ~60fps time step for demo
    }

    // Cleanup
    cudaFree(d_pixels);
    cudaGraphicsUnregisterResource(cudaTexRes);

    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
