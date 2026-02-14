// main.cu
#include <cstdio>
#include <vector>
#include <cmath>

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

__global__ void renderKernel(uchar4* out, int w, int h, float t) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    float u = (x + 0.5f) / (float)w;
    float v = (y + 0.5f) / (float)h;

    // Simple animated gradient (proof the pipeline is alive)
    float r = u;
    float g = v;
    float b = 0.5f + 0.5f * sinf(t + u * 6.2831f);

    unsigned char R = (unsigned char)(255.0f * fminf(fmaxf(r, 0.0f), 1.0f));
    unsigned char G = (unsigned char)(255.0f * fminf(fmaxf(g, 0.0f), 1.0f));
    unsigned char B = (unsigned char)(255.0f * fminf(fmaxf(b, 0.0f), 1.0f));

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

        // CUDA render
        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        renderKernel << <grid, block >> > (d_pixels, W, H, t);
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
