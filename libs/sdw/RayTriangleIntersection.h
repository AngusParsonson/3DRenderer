#include <glm/glm.hpp>
#include <iostream>

class RayTriangleIntersection
{
  public:
    glm::vec3 intersectionPoint;
    float distanceFromCamera;
    ModelTriangle intersectedTriangle;
    Colour colour;

    RayTriangleIntersection()
    {
    }

    RayTriangleIntersection(glm::vec3 point, float distance, ModelTriangle triangle, Colour col)
    {
        intersectionPoint = point;
        distanceFromCamera = distance;
        intersectedTriangle = triangle;
	colour = col;
    }
};

/*std::ostream& operator<<(std::ostream& os, const RayTriangleIntersection& intersection)
{
    os << "Intersection is at " << intersection.intersectionPoint << " on triangle " << intersection.intersectedTriangle << " at a distance of " << intersection.distanceFromCamera << std::endl;
    return os;
}*/
