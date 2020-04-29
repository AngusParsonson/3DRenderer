#include <glm/glm.hpp>
#include "Colour.h"
#include <string>

class ModelTriangle
{
  public:
    glm::vec3 vertices[3];
    Colour colour;
    glm::vec2 textureVertices[3];
    int textureFileIndex;

    ModelTriangle()
    {
    }

    ModelTriangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, Colour trigColour)
    {
      vertices[0] = v0;
      vertices[1] = v1;
      vertices[2] = v2;
      textureVertices[0] = glm::vec2(-1 ,-1);
      colour = trigColour;
      textureFileIndex = -1;
    }

    ModelTriangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec2 vt0, glm::vec2 vt1, glm::vec2 vt2, int fileInd) {
      vertices[0] = v0;
      vertices[1] = v1;
      vertices[2] = v2;
      textureVertices[0] = vt0;
      textureVertices[1] = vt1;
      textureVertices[2] = vt2;
      colour = Colour(255, 255, 255);
      textureFileIndex = fileInd;
    }
};

std::ostream& operator<<(std::ostream& os, const ModelTriangle& triangle)
{
    os << "(" << triangle.vertices[0].x << ", " << triangle.vertices[0].y << ", " << triangle.vertices[0].z << ")" << std::endl;
    os << "(" << triangle.vertices[1].x << ", " << triangle.vertices[1].y << ", " << triangle.vertices[1].z << ")" << std::endl;
    os << "(" << triangle.vertices[2].x << ", " << triangle.vertices[2].y << ", " << triangle.vertices[2].z << ")" << std::endl;
    os << "(" << triangle.colour.red << ", " << triangle.colour.green << ", " << triangle.colour.blue << ")" << std::endl;
    os << std::endl;
    return os;
}