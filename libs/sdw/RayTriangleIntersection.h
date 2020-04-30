#include <glm/glm.hpp>
#include <iostream>

class RayTriangleIntersection
{
  public:
    glm::vec3 intersectionPoint;
    glm::vec3 localIntersection;
    ModelTriangle intersectedTriangle;

    RayTriangleIntersection()
    {
    }

    RayTriangleIntersection(glm::vec3 point, glm::vec3 local, ModelTriangle triangle)
    {
        intersectionPoint = point;
        localIntersection = local;
        intersectedTriangle = triangle;
    }
};

/*std::ostream& operator<<(std::ostream& os, const RayTriangleIntersection& intersection)
{
    os << "Intersection is at " << intersection.intersectionPoint << " on triangle " << intersection.intersectedTriangle << " at a distance of " << intersection.distanceFromCamera << std::endl;
    return os;
}*/
