// Manuel Correia Neves Tomas da Silva 2024
// -----------------------------------------------------------------------------
// I was recently given the challenge to explain How to:
//	Given a plane defined by three points (triangle) and an arbitrary point
//	Find the closest point in the plane to the arbitrary point.
// 
// Unfortunately I was not able to express myself fully at the time.
// So I wrote this to redeem myself as an exercise
// 
// First given the 3 points A, B, C we find the normal that defines the plane
// we get the vectors BA and CA and with the cross product find the normal N 
// 
// We then find the vector between A and our arbitrary point P (PA)
// This is not the vector we want since its "tilted"
// But by doing dot product with it and the plane normal N 
// we can find the magnitude of our projection vector.
// 
// with that magnitude we found we can multiply it to the normal and flip its direction to get the projection vector.
// 
// by adding this new vector to the arbitrary point we found the point in the plane we were looking for 
//
// In this implementation I did a bit of an extra challenge
// and after getting the point in the plane 
// I then find the projected point in the triangle we started with
// 
// To achieve this first we check if we are already in the triangle
// for this we make three vectors by subtracting A B C with our new projected point X
// and we get AX, BX, CX by using cross product between CX/BX, CX/AX, AX/BX
// and we check with the dot product if their directions are alligned (> 0)
// if so we are inside the triangle and our work is already done
// otherwise we are outside and if so we do a similar projection as before but for each side of the triangle
// and from the projected points we get we check the closest projected point
// with a simple distance check.
// 
// You can find everything described in the getClosestPointOnModel function
// 
// -----------------------------------------------------------------------------
// TODO: 
// 
// (not enouughh time / outside of the current scope)
// - Improve the code layout, currently everything is one big file
// - Abstract the idea of Shader, lots of repeating code and constant unnecessary set up and destruction of shaders
// - Cache the cameraPosition from getCameraPositionFromView and only calculate it if it gets dirty
// - make window resizable
// - Use a BVH data structure to avoid calculations on highpoly meshes
// - dont create and delete the dots, grids and lines each frame (its not affecting performance currently but its waste)
// - make the calculation on the shader for the fun of it

#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>


namespace mySimpleModelViewer
{
#pragma region Classes
	// Add UserSettings Here
	struct UserSettings
	{
		bool autoRotate = false;
		bool showRotationGizmo = false;

		bool calculateClosestPoint = true;

		bool renderCurrentVertices = false;
		bool renderClosestPointInMesh = false;
		bool renderMeshProjectionVector = false;
		bool renderMeshProjectionAuxiliaryVector = false;

		bool renderClosestPointInPlane = false;
		bool renderPlaneProjectionVector = false;
		bool renderPlaneProjectionAuxiliaryVector = false;
		bool renderPlane = false;

		bool followClosestPoint = false;
		bool closestPointNearCamera = false;

		bool displayMeshInsteadOfTriangle = false;
	};

	class ScopedSetUp
	{
	public:
		GLFWwindow* window;

	private:
		void init()
		{
			if (!glfwInit())
			{
				std::cerr << "Failed to initialize GLFW" << std::endl;
				return;
			}

			window = glfwCreateWindow(800.0f*2, 600.0f*2, "Simple Model Viewer", nullptr, nullptr);
			if (!window)
			{
				std::cerr << "Failed to create GLFW window" << std::endl;
				glfwTerminate();
				return;
			}

			glfwMakeContextCurrent(window);
			glewInit();

			IMGUI_CHECKVERSION();
			ImGui::CreateContext();

			{
				ImGuizmo::Style& style = ImGuizmo::GetStyle();
				style.RotationLineThickness = 10;
			}

			ImGui_ImplGlfw_InitForOpenGL(window, true);
			ImGui_ImplOpenGL3_Init("#version 330");
		}

		void cleanUp()
		{
			glfwDestroyWindow(window);
			glfwTerminate();
		}

	public:
		ScopedSetUp()
		{
			init();
		}

		~ScopedSetUp()
		{
			cleanUp();
		}
	};

	class Camera
	{
	public:
		glm::mat4 view;
		glm::mat4 perspective;
		float depth;

		glm::vec3 getCameraPositionFromView() const
		{
			glm::mat4 invView = glm::inverse(view);
			glm::vec3 cameraPos = glm::vec3(invView[3]);
			cameraPos = glm::normalize(cameraPos);
			cameraPos *= fabs(depth);
			return cameraPos;
		}

		glm::mat4 getPerspectiveView() const
		{
			return perspective * view;
		}
	};

	class Grid
	{
		std::vector<GLfloat> vertices;
		glm::vec3 position;
		glm::vec2 size;
		glm::vec3 normal;

		int shaderProgram;
		GLuint VBO;
		GLuint VAO;
		GLuint EBO;

	public:
		Grid(glm::vec3 pos, glm::vec3 normalDir, glm::vec2 quadSize = glm::vec2(1.0f, 1.0f))
			: position(pos), normal(glm::normalize(normalDir)), size(quadSize)
		{
			glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
			if (glm::abs(glm::dot(up, normal)) > 0.999f)
			{
				// Handle the case where the normal is (almost) parallel to the up vector
				up = glm::vec3(1.0f, 0.0f, 0.0f);
			}

			// Calculate tangent and bitangent 
			glm::vec3 tangent = glm::normalize(glm::cross(up, normal));
			glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

			vertices = {
				// Positions & Texture Coords
				position.x + (-tangent.x - bitangent.x) * size.x / 2.0f,
				position.y + (-tangent.y - bitangent.y) * size.y / 2.0f,
				position.z + (-tangent.z - bitangent.z) * size.x / 2.0f,  0.0f, 0.0f, // Bottom-left

				position.x + (tangent.x - bitangent.x) * size.x / 2.0f,
				position.y + (tangent.y - bitangent.y) * size.y / 2.0f,
				position.z + (tangent.z - bitangent.z) * size.x / 2.0f,  1.0f, 0.0f, // Bottom-right

				position.x + (tangent.x + bitangent.x) * size.x / 2.0f,
				position.y + (tangent.y + bitangent.y) * size.y / 2.0f,
				position.z + (tangent.z + bitangent.z) * size.x / 2.0f,  1.0f, 1.0f, // Top-right

				position.x + (-tangent.x + bitangent.x) * size.x / 2.0f,
				position.y + (-tangent.y + bitangent.y) * size.y / 2.0f,
				position.z + (-tangent.z + bitangent.z) * size.x / 2.0f,  0.0f, 1.0f  // Top-left
			};

			GLuint indices[] = {
				0, 1, 2,   // First triangle
				2, 3, 0    // Second triangle
			};

			// Shader setup (same as before)
			const char* vertexShaderSource = R"(
				#version 330 core
				layout(location = 0) in vec3 aPos;
				layout(location = 1) in vec2 aTexCoord;

				out vec2 TexCoord;

				uniform mat4 MVP;

				void main() {
					gl_Position = MVP * vec4(aPos, 1.0);
					TexCoord = aTexCoord;
				}
				)";

			const char* fragmentShaderSource = R"(
				#version 330 core
				in vec2 TexCoord;
				out vec4 FragColor;

				uniform vec2 gridSize;
				uniform vec4 gridColor;
				uniform vec4 backgroundColor;

				void main() {
					float lineThickness = 0.01;
					vec2 gridLines = abs(fract(TexCoord * gridSize * gridSize) - 0.5);

					if (min(gridLines.x, gridLines.y) < lineThickness) {
						FragColor = backgroundColor;
					} else {
						FragColor = gridColor;
					}
				}
				)";

			// Compile and link shaders
			int vertexShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
			glCompileShader(vertexShader);
			GLint success;
			GLchar infoLog[512];
			glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
			glCompileShader(fragmentShader);
			if (!success)
			{
				glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
			}


			shaderProgram = glCreateProgram();
			glAttachShader(shaderProgram, vertexShader);
			glAttachShader(shaderProgram, fragmentShader);
			glLinkProgram(shaderProgram);
			GLint linkStatus;
			glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkStatus);
			if (!linkStatus)
			{
				glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			}

			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);

			// Generate and bind buffers
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glGenBuffers(1, &EBO);

			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

			// Position attribute
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			// Texture coordinate attribute
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		void render(const Camera& camera)
		{
			glUseProgram(shaderProgram);
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, &camera.getPerspectiveView()[0][0]);

			// Set grid properties
			glUniform2f(glGetUniformLocation(shaderProgram, "gridSize"), 10.0f, 10.0f);
			glUniform4f(glGetUniformLocation(shaderProgram, "gridColor"), 0.0f, 0.0f, 0.0f, 0.5f);  // Black grid, 50% transparent
			glUniform4f(glGetUniformLocation(shaderProgram, "backgroundColor"), 1.0f, 1.0f, 1.0f, 0.2f);  // White background, 20% transparent

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
			//glDisable(GL_BLEND);

		}

		~Grid()
		{
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);
			glDeleteBuffers(1, &EBO);
			glDeleteProgram(shaderProgram);
		}
	};

	class Point
	{
		std::vector<GLfloat> vertices;
		glm::vec3 position;
		glm::vec3 pointColor;
		float pointSize;

		int shaderProgram;
		GLuint VBO;
		GLuint VAO;

	public:
		Point(glm::vec3 pos, float size = 5.0f, glm::vec3 color = glm::vec3(1, 1, 1))
			: position(pos), pointSize(size), pointColor(color)
		{
			const char* vertexShaderSource = R"(
					#version 330 core
					layout(location = 0) in vec3 aPos;

					uniform mat4 MVP;

					void main() {
						gl_Position = MVP * vec4(aPos.x, aPos.y, aPos.z, 1.0);
						gl_PointSize = 5.0;
					}
				)";

			const char* fragmentShaderSource = R"(
					#version 330 core
					out vec4 FragColor;

					uniform vec3 color;

					void main() {
						FragColor = vec4(color, 1.0f);
					}
				)";

			int vertexShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
			glCompileShader(vertexShader);
			GLint success;
			GLchar infoLog[512];
			glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
			glCompileShader(fragmentShader);
			if (!success)
			{
				glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			shaderProgram = glCreateProgram();
			glAttachShader(shaderProgram, vertexShader);
			glAttachShader(shaderProgram, fragmentShader);
			glLinkProgram(shaderProgram);
			GLint linkStatus;
			glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkStatus);
			if (!linkStatus)
			{
				glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			}

			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);

			vertices = {
				position.x,
				position.y,
				position.z
			};

			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		void render(const Camera& camera)
		{
			glUseProgram(shaderProgram);
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, &camera.getPerspectiveView()[0][0]);
			glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &pointColor[0]);

			glPointSize(pointSize);

			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, 1);
			glPointSize(1.0f);
		}

		~Point()
		{
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);
			glDeleteProgram(shaderProgram);
		}
	};

	class Line
	{
		std::vector<GLfloat> vertices;
		glm::vec3 startPoint;
		glm::vec3 endPoint;
		glm::vec3 lineColor;

		float lineWidth;

		int shaderProgram;
		GLuint VBO;
		GLuint VAO;

	public:

		Line(glm::vec3 start, glm::vec3 end, float width = 1.0f, glm::vec3 color = glm::vec3(1, 1, 1)) :
			startPoint(start), endPoint(end), lineWidth(width), lineColor(color)
		{
			const char* vertexShaderSource = R"(
					#version 330 core
					layout(location = 0) in vec3 aPos;

					uniform mat4 MVP;

					void main() {
						gl_Position = MVP * vec4(aPos.x, aPos.y, aPos.z, 1.0);
					}
					)";

			const char* fragmentShaderSource = R"(
					#version 330 core
					out vec4 FragColor;

					uniform vec3 color;

					void main() {
						FragColor = vec4(color, 1.0f);
					}
					)";


			int vertexShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
			glCompileShader(vertexShader);
			GLint success;
			GLchar infoLog[512];
			glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
			glCompileShader(fragmentShader);
			if (!success)
			{
				glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			shaderProgram = glCreateProgram();
			glAttachShader(shaderProgram, vertexShader);
			glAttachShader(shaderProgram, fragmentShader);
			glLinkProgram(shaderProgram);
			GLint linkStatus;
			glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkStatus);
			if (!linkStatus)
			{
				glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			}

			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);

			vertices =
			{
				 start.x,
				 start.y,
				 start.z,
				 end.x,
				 end.y,
				 end.z,
			};

			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);

		}

		void render(const Camera& camera)
		{
			glUseProgram(shaderProgram);
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, &camera.getPerspectiveView()[0][0]);
			glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &lineColor[0]);


			glLineWidth(lineWidth);
			glBindVertexArray(VAO);
			glDrawArrays(GL_LINES, 0, 2);
			glLineWidth(1.0f);
		}

		~Line()
		{
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);
			glDeleteProgram(shaderProgram);
		}
	};

	class Model
	{
	public:
		std::vector<GLfloat> verticesAndNormals;
		std::vector<GLuint> indices;
		glm::mat4 TRS;

	private:
		const char* modelPath;
		float modelImportScale;

		GLuint VAO;
		GLuint VBO;
		GLuint EBO;

		GLuint vertexShader;
		GLuint fragmentShader;
		GLuint shaderProgram;

		void loadModel()
		{
			Assimp::Importer importer;
			const aiScene* scene = importer.ReadFile(modelPath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

			if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
			{
				std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
				return;
			}

			for (unsigned int i = 0; i < scene->mNumMeshes; i++)
			{
				aiMesh* mesh = scene->mMeshes[i];

				for (unsigned int j = 0; j < mesh->mNumVertices; j++)
				{
					glm::vec3 vertex(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
					glm::vec3 normal(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
					verticesAndNormals.push_back(vertex.x * modelImportScale);
					verticesAndNormals.push_back(vertex.y * modelImportScale);
					verticesAndNormals.push_back(vertex.z * modelImportScale);

					verticesAndNormals.push_back(normal.x);
					verticesAndNormals.push_back(normal.y);
					verticesAndNormals.push_back(normal.z);
				}

				for (unsigned int j = 0; j < mesh->mNumFaces; j++)
				{
					aiFace face = mesh->mFaces[j];
					for (unsigned int k = 0; k < face.mNumIndices; k++)
					{
						indices.push_back(face.mIndices[k]);
					}
				}
			}
		}

		void setUpShaders()
		{
			const char* vertexShaderSource = R"(
					#version 330 core
					layout(location = 0) in vec3 aPos;
					layout(location = 1) in vec3 aNormal;

					uniform mat4 model;
					uniform mat4 view;
					uniform mat4 projection;

					out vec3 FragPos;
					flat out vec3 Normal; // needs to match fragment shader on the 3060

					void main() {
						FragPos = vec3(model * vec4(aPos, 1.0));
						Normal = mat3(transpose(inverse(model))) * aNormal;
						gl_Position = projection * view * model * vec4(aPos, 1.0);
					}
					)";

			const char* fragmentShaderSource = R"(
					#version 330 core
					out vec4 FragColor;

					in vec3 FragPos;
					flat in vec3 Normal;  // flat makes it easier to see what Im trying to show

					uniform vec3 lightPos;
					uniform vec3 lightColor;
					uniform vec3 objectColor;

					void main() {
						float ambientStrength = 0.2;
						vec3 ambient = ambientStrength * lightColor;

						vec3 norm = normalize(Normal);
						vec3 lightDir = normalize(lightPos - FragPos);
						float diff = max(dot(norm, lightDir), 0.0);
						vec3 diffuse = diff * lightColor;

						vec3 result = (ambient + diffuse) * objectColor;
						FragColor = vec4(result, 1.0);
					})";


			vertexShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
			glCompileShader(vertexShader);
			GLint success;
			GLchar infoLog[512];
			glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
			glCompileShader(fragmentShader);
			glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
			}

			shaderProgram = glCreateProgram();
			glAttachShader(shaderProgram, vertexShader);
			glAttachShader(shaderProgram, fragmentShader);
			glLinkProgram(shaderProgram);
			GLint linkStatus;
			glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkStatus);
			if (!linkStatus)
			{
				glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			}
		}

		void prepareBuffers()
		{
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glGenBuffers(1, &EBO);

			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, verticesAndNormals.size() * sizeof(GLfloat), verticesAndNormals.data(), GL_STATIC_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)0);
			glEnableVertexAttribArray(0);

			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
			glEnableVertexAttribArray(1);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

	public:
		Model() : modelPath(NULL), modelImportScale(1.0f), TRS(1)
		{
			// if used need to call setUp()!!
		}

		Model(const char* path, float importScale = 1.0f) : modelPath(path), modelImportScale(importScale), TRS(1)
		{
			setUpShaders();
			loadModel();
			prepareBuffers();
		}

		void setUp()
		{
			setUpShaders();
			prepareBuffers();
		}

		void render(const Camera& camera)
		{
			glUseProgram(shaderProgram);

			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(TRS));
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(camera.view));
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(camera.perspective));

			glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 10.0f, 1.0f, 1.0f);
			glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.5f, 0.31f);

			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
		}

		~Model()
		{
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);
			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);
			glDeleteProgram(shaderProgram);
		}
	};
#pragma endregion //~Classes

	UserSettings _settings;
	glm::vec3 _arbitraryPoint(10, 0, 0);

	glm::vec3 getClosestPointOnModel(const glm::vec3& point, const Model& model, const Camera& camera);
	void runApplication();
	void processInput(GLFWwindow* window, Camera& camera, float deltaTime);

	void renderDebugData(const Model& model, int bestIndex, const Camera& camera, const glm::vec3& point, glm::vec3 discoveredPoint);
	void renderArrow(const glm::vec3& startPos, const glm::vec3& endPos, const glm::vec3& color, const Camera& camera);
	void renderDot(const glm::vec3& position, const glm::vec3& color, const Camera& camera);
	void renderBackground(const Camera& camera);
	void renderGizmo(glm::mat4& modelMatrix, Camera& camera);
	void renderImGUI(float deltaTime, glm::mat4& modelMatrix, Camera& camera);
}

void main() { mySimpleModelViewer::runApplication(); }

namespace mySimpleModelViewer
{
	glm::vec3 getClosestPointOnModel(const glm::vec3& point, const Model& model, const Camera& camera)
	{
		// if its pointing up it failed
		glm::vec3 discoveredPoint(0, 10, 0);

		float closestValue = std::numeric_limits<float>::max();
		int bestIndex = 0;

		for (auto i = 0; i < model.indices.size() - 2; i += 3)
		{
			int i_p1 = model.indices[i + 0];
			int i_p2 = model.indices[i + 1];
			int i_p3 = model.indices[i + 2];

			const float* vertData = &model.verticesAndNormals[0];
			glm::vec3 p1(vertData[6 * i_p1], vertData[6 * i_p1 + 1], vertData[6 * i_p1 + 2]);
			glm::vec3 p2(vertData[6 * i_p2], vertData[6 * i_p2 + 1], vertData[6 * i_p2 + 2]);
			glm::vec3 p3(vertData[6 * i_p3], vertData[6 * i_p3 + 1], vertData[6 * i_p3 + 2]);

			// apply model TRS to the points
			p1 = model.TRS * glm::vec4(p1, 0);
			p2 = model.TRS * glm::vec4(p2, 0);
			p3 = model.TRS * glm::vec4(p3, 0);

			glm::vec3 v1 = p2 - p1;
			glm::vec3 v2 = p3 - p1;

			glm::vec3 planeNormal = glm::cross(v1, v2);
			planeNormal = glm::normalize(planeNormal);

			float distanceToPlane = glm::dot(planeNormal, (point - p1));
			distanceToPlane *= -1;

			glm::vec3 translationVector = planeNormal * distanceToPlane;
			glm::vec3 closestPointOnThePlane = point + translationVector;

			// closestPointOnThePlane is the projection on the plane defined by the 3 vertices
			// now make sure the point is inside the triangle
			glm::vec3 _p1 = p1 - closestPointOnThePlane;
			glm::vec3 _p2 = p2 - closestPointOnThePlane;
			glm::vec3 _p3 = p3 - closestPointOnThePlane;

			glm::vec3 u = glm::cross(_p2, _p3);
			glm::vec3 v = glm::cross(_p3, _p1);
			glm::vec3 w = glm::cross(_p1, _p2);

			float uvAngle = glm::dot(u, v);
			float uwAngle = glm::dot(u, w);

			// if any is negative the value is outside the triangle
			// need to do a second projection
			// find the projected point for each and see which one is closest

			if (uvAngle < 0.0f || uwAngle < 0.0f)
			{

				// could extract to a function but keeping it here to vizualize the math better
				glm::vec3 closestPointOnTheEdge1;
				{
					glm::vec3 edge = p2 - p1;
					float t = glm::dot(closestPointOnThePlane - p1, edge) / glm::dot(edge, edge);
					t = glm::clamp(t, 0.0f, 1.0f);
					closestPointOnTheEdge1 = p1 + t * edge;
				}

				glm::vec3 closestPointOnTheEdge2;
				{
					glm::vec3 edge = p3 - p2;
					float t = glm::dot(closestPointOnThePlane - p2, edge) / glm::dot(edge, edge);
					t = glm::clamp(t, 0.0f, 1.0f);
					closestPointOnTheEdge2 = p2 + t * edge;
				}

				glm::vec3 closestPointOnTheEdge3;
				{
					glm::vec3 edge = p1 - p3;
					float t = glm::dot(closestPointOnThePlane - p3, edge) / glm::dot(edge, edge);
					t = glm::clamp(t, 0.0f, 1.0f);
					closestPointOnTheEdge3 = p3 + t * edge;
				}

				// Choose the closest of these points
				float dist1 = glm::length(closestPointOnTheEdge1 - closestPointOnThePlane);
				float dist2 = glm::length(closestPointOnTheEdge2 - closestPointOnThePlane);
				float dist3 = glm::length(closestPointOnTheEdge3 - closestPointOnThePlane);

				if (dist1 < dist2 && dist1 < dist3)
				{
					closestPointOnThePlane = closestPointOnTheEdge1;
				}
				else if (dist2 < dist3)
				{
					closestPointOnThePlane = closestPointOnTheEdge2;
				}
				else
				{
					closestPointOnThePlane = closestPointOnTheEdge3;
				}
			}

			float distance = glm::distance(closestPointOnThePlane, point);

			if (distance < closestValue)
			{
				discoveredPoint = closestPointOnThePlane;
				closestValue = distance;
				bestIndex = i;
			}
		}

		// render things for debug
		renderDebugData(model, bestIndex, camera, point, discoveredPoint);

		return discoveredPoint;
	}

	void runApplication()
	{
		// Sets up Glew, GLFW, IMGui and ImGuizmo
		// cleans up on destruction
		ScopedSetUp setUp;

		// tested .obj, .fbx, glb
		Model meshModel("meshes/suzzane.obj", 5.0f);
		Model triangle;
		{
			glm::vec3 p1(1.0f, 0.0f, 0.0f);
			glm::vec3 p2(1.0f, 1.0f, 0.0f);
			glm::vec3 p3(0.0f, 0.0f, 1.0f);

			float scale = 5.0f;

			p1 *= scale;
			p2 *= scale;
			p3 *= scale;

			glm::vec3 v1 = p2 - p1;
			glm::vec3 v2 = p3 - p1;

			glm::vec3 n = glm::cross(v1, v2);
			n = glm::normalize(n);

			auto vertices = &triangle.verticesAndNormals;
			vertices->emplace_back(p1.x);
			vertices->emplace_back(p1.y);
			vertices->emplace_back(p1.z);
			vertices->emplace_back(n.x);
			vertices->emplace_back(n.y);
			vertices->emplace_back(n.z);

			vertices->emplace_back(p2.x);
			vertices->emplace_back(p2.y);
			vertices->emplace_back(p2.z);
			vertices->emplace_back(n.x);
			vertices->emplace_back(n.y);
			vertices->emplace_back(n.z);

			vertices->emplace_back(p3.x);
			vertices->emplace_back(p3.y);
			vertices->emplace_back(p3.z);
			vertices->emplace_back(n.x);
			vertices->emplace_back(n.y);
			vertices->emplace_back(n.z);

			auto indices = &triangle.indices;
			indices->emplace_back(0);
			indices->emplace_back(1);
			indices->emplace_back(2);

			triangle.setUp();
		}

		//Model& model = meshModel;

		Camera camera;
		{
			camera.depth = -25.0f;
			float startAngle = 35.0f;
			float camX = sin(startAngle) * camera.depth;
			float camZ = cos(startAngle) * camera.depth;
			glm::vec3 cameraStartPos(camX, camX, camZ);

			camera.view = glm::lookAt(cameraStartPos, { 0,0,0 }, glm::vec3(0.0f, 1.0f, 0.0f));
			camera.perspective = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.01f, 10000000.0f);
		}

		double previousFrame = 0.0;
		GLFWwindow* window = setUp.window;
		while (!glfwWindowShouldClose(window))
		{
			double currentFrame = glfwGetTime();
			double deltaTime = currentFrame - previousFrame;
			previousFrame = currentFrame;

			Model& model = _settings.displayMeshInsteadOfTriangle ? meshModel : triangle;

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);

			model.render(camera);
			glm::vec3 closestPoint(0);
			if (_settings.calculateClosestPoint)
			{
				closestPoint = getClosestPointOnModel(_arbitraryPoint, model, camera);
			}

			if (!_settings.renderPlane || !_settings.calculateClosestPoint)
			{
				renderBackground(camera);
			}

			// render last but before any new user inputs
			renderImGUI(deltaTime, model.TRS, camera);

			processInput(window, camera, deltaTime);

			glm::vec3 target = glm::vec3(0, 0, 0);
			glm::vec3 startPosition = camera.getCameraPositionFromView();;
			if (_settings.followClosestPoint)
			{
				target = closestPoint;
				startPosition = _arbitraryPoint;
				startPosition += glm::vec3(0, 2, 0);
			}

			camera.view = glm::lookAt(startPosition, target, glm::vec3(0.0f, 1.0f, 0.0f));

			if (_settings.autoRotate)
			{
				model.TRS = glm::rotate_slow(model.TRS, static_cast<float>(deltaTime) * .3f, glm::vec3(1, 1.0f, 0.0f));
			}

			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}

	void renderArrow(const glm::vec3& startPos, const glm::vec3& endPos, const glm::vec3& color, const Camera& camera)
	{
		glm::vec3 destination = endPos - startPos;

		glm::vec3 lineStart = glm::vec3(0, 0, 0);
		glm::vec3 lineEnd = glm::vec3(0, 1, 0);
		glm::vec3 v = lineEnd - lineStart;

		glm::vec3 whisker1Start = glm::vec3(0, 1, 0);
		glm::vec3 whisker1End = glm::vec3(-1.0f / 15.0f, 7.0f / 8.0f, 0);

		glm::vec3 whisker2Start = glm::vec3(0, 1, 0);
		glm::vec3 whisker2End = glm::vec3(1.0f / 15.0f, 7.0f / 8.0f, 0);

		glm::vec3 src = glm::normalize(v);
		glm::vec3 dest = glm::normalize(destination);
		glm::vec3 rotationAxis = glm::cross(src, dest);

		float dotProduct = glm::dot(src, dest);

		float angle = glm::acos(glm::clamp(dotProduct, -1.0f, 1.0f));
		glm::quat rotationQuat = glm::angleAxis(angle, glm::normalize(rotationAxis));

		float length = glm::length(destination);

		lineStart *= length;
		lineEnd *= length;

		whisker1Start *= length;
		whisker1End *= length;

		whisker2Start *= length;
		whisker2End *= length;

		lineStart = rotationQuat * glm::vec4(lineStart, 1);
		lineEnd = rotationQuat * glm::vec4(lineEnd, 1);

		whisker1Start = rotationQuat * glm::vec4(whisker1Start, 1);
		whisker1End = rotationQuat * glm::vec4(whisker1End, 1);

		whisker2Start = rotationQuat * glm::vec4(whisker2Start, 1);
		whisker2End = rotationQuat * glm::vec4(whisker2End, 1);

		lineStart += startPos;
		lineEnd += startPos;

		whisker1Start += startPos;
		whisker1End += startPos;

		whisker2Start += startPos;
		whisker2End += startPos;

		Line line(lineStart, lineEnd, 4.0f, color);
		Line whisker1(whisker1Start, whisker1End, 4.0f, color);
		Line whisker2(whisker2Start, whisker2End, 4.0f, color);

		line.render(camera);
		whisker1.render(camera);
		whisker2.render(camera);
	}

	void renderDot(const glm::vec3& position, const glm::vec3& color, const Camera& camera)
	{
		Point point(position, 15.0f, color);
		point.render(camera);
	}

	void renderBackground(const Camera& camera)
	{
		glm::vec3 cameraPosition = camera.getCameraPositionFromView();
		float y = cameraPosition.y;
		if (y > -10)
		{
			y = -10;
		}

		// make sure the background stays in the background
		Grid backgroundGrid({ 0, y, 0 }, { 0,1,0 }, { 1000 , 1000 });
		backgroundGrid.render(camera);
	}

	void renderDebugData(const Model& model, int bestIndex, const Camera& camera, const glm::vec3& point, glm::vec3 discoveredPoint)
	{
		int i_p1 = model.indices[bestIndex + 0];
		int i_p2 = model.indices[bestIndex + 1];
		int i_p3 = model.indices[bestIndex + 2];

		const float* vertData = &model.verticesAndNormals[0];
		glm::vec3 p1(vertData[6 * i_p1], vertData[6 * i_p1 + 1], vertData[6 * i_p1 + 2]);
		glm::vec3 p2(vertData[6 * i_p2], vertData[6 * i_p2 + 1], vertData[6 * i_p2 + 2]);
		glm::vec3 p3(vertData[6 * i_p3], vertData[6 * i_p3 + 1], vertData[6 * i_p3 + 2]);

		// to avoid clipping with the mesh
		{
			glm::vec3 n1(vertData[6 * i_p1 + 3], vertData[6 * i_p1 + 4], vertData[6 * i_p1 + 5]);
			glm::vec3 n2(vertData[6 * i_p2 + 3], vertData[6 * i_p2 + 4], vertData[6 * i_p2 + 5]);
			glm::vec3 n3(vertData[6 * i_p3 + 3], vertData[6 * i_p3 + 4], vertData[6 * i_p3 + 5]);

			float adjustment = 0.1f;
			p1 += n1 * adjustment;
			p2 += n2 * adjustment;
			p3 += n3 * adjustment;
		}

		p1 = model.TRS * glm::vec4(p1, 0);
		p2 = model.TRS * glm::vec4(p2, 0);
		p3 = model.TRS * glm::vec4(p3, 0);

		if (_settings.renderCurrentVertices)
		{
			renderDot(p1, { 1,1,1 }, camera);
			renderDot(p2, { 1,1,1 }, camera);
			renderDot(p3, { 1,1,1 }, camera);
		}

		glm::vec3 v1 = p2 - p1;
		glm::vec3 v2 = p3 - p1;

		glm::vec3 planeNormal = glm::cross(v1, v2);
		planeNormal = glm::normalize(planeNormal);


		if (_settings.renderPlaneProjectionAuxiliaryVector)
		{
			renderArrow(p1, p2, { 1,0,0 }, camera);
			renderArrow(p1, p3, { 0,1,0 }, camera);
			renderArrow(p1, p1 + planeNormal, { 0,0,1 }, camera);
		}

		float distanceToPlane = glm::dot(planeNormal, (point - p1));
		distanceToPlane *= -1;

		glm::vec3 translationVector = planeNormal * distanceToPlane;
		glm::vec3 closestPointOnThePlane = point + translationVector;

		if (_settings.renderClosestPointInPlane)
		{
			renderDot(closestPointOnThePlane, { 1,0,1 }, camera);
		}

		if (_settings.renderPlaneProjectionVector)
		{
			Line line(closestPointOnThePlane, point, 4.0f);
			line.render(camera);
		}

		{
			glm::vec3 _p1 = p1 - closestPointOnThePlane;
			glm::vec3 _p2 = p2 - closestPointOnThePlane;
			glm::vec3 _p3 = p3 - closestPointOnThePlane;

			glm::vec3 u = glm::cross(_p2, _p3);
			glm::vec3 v = glm::cross(_p3, _p1);
			glm::vec3 w = glm::cross(_p1, _p2);

			u = glm::normalize(u);
			v = glm::normalize(v);
			w = glm::normalize(w);

			if (_settings.renderMeshProjectionAuxiliaryVector)
			{
				Line line1(closestPointOnThePlane, p1, 4, { 1,0,0 });
				line1.render(camera);

				Line line2(closestPointOnThePlane, p2, 4, { 0,1,0 });
				line2.render(camera);

				Line line3(closestPointOnThePlane, p3, 4, { 0,0,1 });
				line3.render(camera);


				renderArrow(closestPointOnThePlane, closestPointOnThePlane + u, { 0,1,1 }, camera);
				renderArrow(closestPointOnThePlane, closestPointOnThePlane + v, { 1,0,1 }, camera);
				renderArrow(closestPointOnThePlane, closestPointOnThePlane + w, { 1,1,0 }, camera);
			}

			if (_settings.renderMeshProjectionVector)
			{
				Line line(closestPointOnThePlane, discoveredPoint, 4.0f);
				line.render(camera);
			}
		}

		{
			glm::vec3 _p1 = p1 - closestPointOnThePlane;
			glm::vec3 _p2 = p2 - closestPointOnThePlane;
			glm::vec3 _p3 = p3 - closestPointOnThePlane;

			glm::vec3 u = glm::cross(_p2, _p3);
			glm::vec3 v = glm::cross(_p3, _p1);
			glm::vec3 w = glm::cross(_p1, _p2);

			float uvAngle = glm::dot(u, v);
			float uwAngle = glm::dot(u, w);

			// if any is negative the value is outside the triangle
			// need to do a second projection
			// find the projected point for each and see which one is closest

			if (uvAngle < 0.0f || uwAngle < 0.0f)
			{
				if (_settings.renderClosestPointInMesh)
				{
					renderDot(discoveredPoint, { 0,1,0 }, camera);
				}
			}
			else
			{
				if (_settings.renderClosestPointInMesh)
				{
					renderDot(discoveredPoint, { 1,0,1 }, camera);
				}
			}
		}

		if (_settings.renderPlane)
		{
			renderBackground(camera);

			// TODO cache the results of this function and change it only when it gets dirtied
			glm::vec3 cameraPosition = camera.getCameraPositionFromView();
			float distance = glm::distance(cameraPosition, { 0,0,0 });

			// done to avoid ugly
			float size = 60 + (distance * distance) / 10.0f;
			Grid planeGrid(closestPointOnThePlane - planeNormal * 0.2f, planeNormal, { size , size });
			planeGrid.render(camera);
		}
	}

	void processInput(GLFWwindow* window, Camera& camera, float deltaTime)
	{
		const float cameraSpeed = 20.5f * deltaTime;

		if (_settings.followClosestPoint)
		{
			return;
		}

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.depth += cameraSpeed * 2;
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.depth -= cameraSpeed * 2;
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.view = glm::rotate_slow(camera.view, cameraSpeed, glm::vec3(0, 1, 0));
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.view = glm::rotate_slow(camera.view, -cameraSpeed, glm::vec3(0, 1, 0));
	}

	void renderGizmo(glm::mat4& modelMatrix, Camera& camera)
	{
		ImGuizmo::BeginFrame();
		ImGuizmo::SetOrthographic(false);
		ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());

		float windowWidth = ImGui::GetIO().DisplaySize.x;
		float windowHeight = ImGui::GetIO().DisplaySize.y;
		ImGuizmo::SetRect(0, 0, windowWidth, windowHeight);

		glm::mat4x4 cube(1);
		cube = glm::translate(cube, _arbitraryPoint);
		cube = glm::scale(cube, glm::vec3(.5f, .5f, .5f));

		if (!_settings.autoRotate && !_settings.followClosestPoint && _settings.showRotationGizmo)
		{
			static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
			ImGuizmo::Manipulate(glm::value_ptr(camera.view), glm::value_ptr(camera.perspective), ImGuizmo::ROTATE, ImGuizmo::LOCAL, glm::value_ptr(modelMatrix), NULL, boundsSnap);
		}

		if (!_settings.followClosestPoint)
		{
			if (_settings.calculateClosestPoint)
			{
				ImGuizmo::DrawCubes(glm::value_ptr(camera.view), glm::value_ptr(camera.perspective), glm::value_ptr(cube), 1);
			}

			ImGuizmo::ViewManipulate(glm::value_ptr(camera.view), 10.0f, { 0,0 }, { 200, 200 }, ImGuizmo::COLOR::DIRECTION_X);
		}


	}

	void renderImGUI(float deltaTime, glm::mat4& modelMatrix, Camera& camera)
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		{
			ImGui::NewFrame();

			renderGizmo(modelMatrix, camera);

			auto flags = ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;

			ImGui::Begin("Debug Data", NULL, flags);
			{
				if (_settings.displayMeshInsteadOfTriangle)
				{
					ImGui::BeginDisabled();
				}

				if (ImGui::Button("Mesh"))
				{
					_settings.displayMeshInsteadOfTriangle = !_settings.displayMeshInsteadOfTriangle;
				}
				else if (_settings.displayMeshInsteadOfTriangle)
				{
					ImGui::EndDisabled();
				}

				ImGui::SameLine();

				if (!_settings.displayMeshInsteadOfTriangle)
				{
					ImGui::BeginDisabled();
				}
				if (ImGui::Button("Triangle"))
				{
					_settings.displayMeshInsteadOfTriangle = !_settings.displayMeshInsteadOfTriangle;
				}
				else if (!_settings.displayMeshInsteadOfTriangle)
				{
					ImGui::EndDisabled();
				}

				ImGui::NewLine();

				ImGui::BeginDisabled();
				auto fps = 1.0f / deltaTime;
				ImGui::DragFloat("FPS", &fps);
				ImGui::EndDisabled();
				ImGui::Checkbox("Auto Rotate", &_settings.autoRotate);

				if (!_settings.autoRotate)
				{
					ImGui::Checkbox("Show Rotation Guizmo", &_settings.showRotationGizmo);
				}

				ImGui::Checkbox("Calculate Closest Point in Mesh", &_settings.calculateClosestPoint);

				ImGui::NewLine();
				ImGui::NewLine();

				if (_settings.calculateClosestPoint)
				{
					ImGui::Checkbox("Look at Closest Point", &_settings.followClosestPoint);
					ImGui::NewLine();

					ImGui::Checkbox("Render Closest Vertices", &_settings.renderCurrentVertices);
					ImGui::Checkbox("Render Plane Projection Aux", &_settings.renderPlaneProjectionAuxiliaryVector);
					ImGui::Checkbox("Render Plane", &_settings.renderPlane);
					ImGui::Checkbox("Render Plane Projection", &_settings.renderPlaneProjectionVector);
					ImGui::Checkbox("Render Point In Plane", &_settings.renderClosestPointInPlane);
					ImGui::NewLine();
					ImGui::Checkbox("Render Point In Mesh", &_settings.renderClosestPointInMesh);
					ImGui::Checkbox("Render Mesh Projection", &_settings.renderMeshProjectionVector);
					ImGui::Checkbox("Render Mesh Projection Aux", &_settings.renderMeshProjectionAuxiliaryVector);
				}
				else
				{
					_settings.followClosestPoint = false;
				}
			}
			ImGui::End();
		}
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}