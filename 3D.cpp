#include <ModelTriangle.h>
#include <CanvasTriangle.h>
#include <RayTriangleIntersection.h>
#include <DrawingWindow.h>
#include <Utils.h>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <bitset>
#include <unordered_map>
#include <algorithm>
#include <math.h>
#include <thread>

using namespace std;
using namespace glm;

#define WIDTH 640
#define HEIGHT 480
#define FOCALLENGTH 300
#define PI 3.14159265
#define INF numeric_limits<double>::infinity()

enum DrawingMode { WIREFRAME, RASTERISE, RAYTRACE};

struct PointLight {
  vec3 position;
  float strength;

  PointLight(vec3 pos, float str) {
    position = pos;
    if (str > 350) strength = 350;
    else strength = str;
  }
};

class Camera {
  private:
    float xTheta = 0;
    float yTheta = 0;
    float zTheta = 0;

    void setRotation() {
      rotation = mat3x3(cos(zTheta), -sin(zTheta), 0,
                        sin(zTheta), cos(zTheta), 0,
                        0, 0, 1) *
                 mat3x3(cos(yTheta), 0, sin(yTheta),
                        0, 1, 0,
                        -sin(yTheta), 0, cos(yTheta)) *
                 mat3x3(1, 0, 0,
                        0, cos(xTheta), -sin(xTheta),
                        0, sin(xTheta), cos(xTheta));
    }

  public:
    vec3 position;
    mat3x3 rotation;
    DrawingMode drawingMode;

    Camera(float x, float y, float depth) {
      position = vec3(x, y, depth);
      drawingMode = WIREFRAME;
      setRotation();
    }

    void translate(float dx, float dy, float ddepth) {
      position.x += dx;
      position.y += dy;
      position.z += ddepth;
    }

    void rotateX(float angle) {
      xTheta += angle * PI / 180;
      setRotation();
    }

    void rotateY(float angle) {
      yTheta += angle * PI / 180;
      setRotation();
    }

    void lookAt() {
      yTheta = tan(position.x / position.z);
      xTheta = -tan(position.y / position.z);
      setRotation();
    }
};

void draw();
void update();
void handleEvent(SDL_Event event);
CanvasTriangle get2DProjection(ModelTriangle modelTriangle, Camera camera);
void drawLine(CanvasPoint from, CanvasPoint to, Colour colour);
void drawStrokedTriangle(CanvasTriangle triangle);
void drawFilledTriangle(CanvasTriangle triangle);
void drawTexturedTriangle(CanvasTriangle triangle, int textureFileIndex);
void drawFilledTrianglesRaytrace(int x0, int x1, int y0, int y1);
void threadRaytrace();
RayTriangleIntersection getClosestIntersection(vec3 rayDirection);
Colour getTextureIntersection(vec3 intersection, ModelTriangle modelTriangle);
bool isInTriangle(vec3 intersectionPoint);
bool isTriangleSelf(ModelTriangle self, ModelTriangle tri);
float calculateBrightness(RayTriangleIntersection intersection);
bool isInShadow(vec3 pointLightVector, float distanceToLight, RayTriangleIntersection self);
vector<CanvasPoint> interpolate2D(CanvasPoint from, CanvasPoint to, float numberOfValues);
vector<CanvasPoint> interpolate2D(TexturePoint from, TexturePoint to, float numberOfValues);
vector<CanvasPoint> interpolate3D(CanvasPoint from, CanvasPoint to, float numberOfValues);
float getLerpNumberOfSteps(CanvasPoint from, CanvasPoint to);
float getLerpNumberOfSteps(TexturePoint from, TexturePoint to);
vector<CanvasTriangle> splitTriangle(CanvasTriangle triangle);
unordered_map<string, Colour> loadMaterial(string file);
void loadOBJ(const char* OBJFile, float scaleFactor);
vector<vector<uint32_t>> loadPPM(string file);
void savePPM();
CanvasPoint getInterpolatedPoint(CanvasPoint maxPoint, CanvasPoint midPoint, CanvasPoint minPoint);
bool isPixelOnScreen(int x, int y);
bool isFlatBotttomedTriangle(CanvasTriangle triangle);
bool clipping(ModelTriangle triangle);

uint32_t BLACK = (255<<24) + (int(0)<<16) + (int(0)<<8) + int(0);
double depthBuffer[WIDTH][HEIGHT] = { { INF } };
int savedPPMs = 0;

DrawingWindow window = DrawingWindow(WIDTH, HEIGHT, false);
Camera camera = Camera(0, 2, 4);
PointLight pointLight = PointLight(vec3(-0.3, 4.8, -3.1), 150);
float ambientBrightness = pointLight.strength / 350;
vector<ModelTriangle> modelTriangles;
vector<CanvasTriangle> canvasTriangles;
vector<vector<vector<uint32_t>>> textureFiles;

int main(int argc, char* argv[])
{
  SDL_Event event;
  loadOBJ("cornell-box.obj", 1);
  loadOBJ("logo.obj", 0.01);
  update();

  while(true)
  {
    // We MUST poll for events - otherwise the window will freeze !
    if(window.pollForInputEvents(&event)) {
      handleEvent(event);
    }

    // Need to render the frame at the end, or nothing actually gets shown on the screen !
    window.renderFrame();
  }
}

void update() {
  window.clearPixels();
  //camera.lookAt();
  vector<CanvasTriangle> newCanvasTriangles;

  for (int i = 0; i < (int)modelTriangles.size(); i++) {
    if (clipping(modelTriangles[i])) newCanvasTriangles.push_back(get2DProjection(modelTriangles[i], camera));   
  }

  canvasTriangles = newCanvasTriangles;
  draw();
}

bool clipping(ModelTriangle triangle) {
  vec3 middle = (triangle.vertices[0] + triangle.vertices[1] + triangle.vertices[2]) / 3.0f;
  vec3 camPos = camera.position * camera.rotation;
  vec3 diff = middle - camPos;

  if (camPos.z - middle.z < 0) return false;
  else if (length(diff) < 1) return false;
  else if (length(diff) > 100) return false;
  else return true;
}

void draw() {
  if(camera.drawingMode == RASTERISE) {
    for (int i = 0; i < (int)canvasTriangles.size(); i++) {
      if (canvasTriangles[i].vertices[0].texturePoint.x == -1) drawFilledTriangle(canvasTriangles[i]);
      else {
        drawTexturedTriangle(canvasTriangles[i], canvasTriangles[i].textureFileIndex);
      }
    }
  }

  else if (camera.drawingMode == WIREFRAME) {
    for (int i = 0; i < (int)canvasTriangles.size(); i++) {
      drawStrokedTriangle(canvasTriangles[i]);
    }
  }

  else if (camera.drawingMode == RAYTRACE) threadRaytrace();
}

// Convert 3D triangle onto canvas space
CanvasTriangle get2DProjection(ModelTriangle modelTriangle, Camera camera) {
  CanvasTriangle projections;
  vector<CanvasPoint> projectionPoints;
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      depthBuffer[x][y] = INF;
    }
  }

  for (int i = 0; i < 3; i++) {
    modelTriangle.vertices[i] = camera.rotation * (modelTriangle.vertices[i] - camera.position);
    float x = ((FOCALLENGTH * (-modelTriangle.vertices[i].x / (modelTriangle.vertices[i].z))) + (WIDTH / 2));
    float y = ((FOCALLENGTH * (modelTriangle.vertices[i].y / (modelTriangle.vertices[i].z))) + (HEIGHT / 2));
    float z = 1/(modelTriangle.vertices[i].z); // Z is inverted so it varies linearly
    CanvasPoint point = CanvasPoint(round(x), round(y), z);
    if (modelTriangle.textureVertices[0].x != -1) point.texturePoint = TexturePoint(modelTriangle.textureVertices[i].x, modelTriangle.textureVertices[i].y);
    projectionPoints.push_back(point);
  }
  
  if (modelTriangle.textureVertices[0].x != -1) return CanvasTriangle(projectionPoints[0], projectionPoints[1], projectionPoints[2], modelTriangle.textureFileIndex);
  else return CanvasTriangle(projectionPoints[0], projectionPoints[1], projectionPoints[2], modelTriangle.colour);
}

void drawLine(CanvasPoint from, CanvasPoint to, Colour colour) {
  float numberOfSteps = getLerpNumberOfSteps(from, to);
  vector<CanvasPoint> points = interpolate3D(from, to, numberOfSteps);

  uint32_t packedColour = (255<<24) + (int(colour.red)<<16) + (int(colour.green)<<8) + int(colour.blue);

  for (float i = 0.0; i < numberOfSteps; i++) {
    float x = round(points[i].x), y = round(points[i].y);
    double z = points[i].depth;

    if (isPixelOnScreen(x, y)) { // Pixel is within resolution
      if(depthBuffer[(int)x][(int)y] > z) { // Pixel is closer to camera than previous pixel
          depthBuffer[(int)x][(int)y] = z;
          window.setPixelColour(x, y, packedColour);
      }
    }
  }
}

// Triangle Rasteriser
// Works by splitting into two flat bottomed tirangles, and then drawing
// lines horizontally across both triangle
void drawFilledTriangle(CanvasTriangle triangle) {
  vector<CanvasTriangle> triangles = splitTriangle(triangle);

  for (int i = 0; i < 2; i++) {
    drawStrokedTriangle(triangles[i]);
    float lineLeftNumberOfSteps = getLerpNumberOfSteps(triangles[i].vertices[0], triangles[i].vertices[1]);
    float lineRightNumberOfSteps = getLerpNumberOfSteps(triangles[i].vertices[0], triangles[i].vertices[2]);
    float numberOfSteps = std::max(lineLeftNumberOfSteps, lineRightNumberOfSteps);

    vector<CanvasPoint> lineLeft  = interpolate3D(triangles[i].vertices[0], triangles[i].vertices[1], numberOfSteps);
    vector<CanvasPoint> lineRight = interpolate3D(triangles[i].vertices[0], triangles[i].vertices[2], numberOfSteps);

    for (size_t j = 0; j < numberOfSteps; j++) {
      drawLine(lineLeft[j], lineRight[j], triangle.colour);
    }
  }
}

void drawTexturedTriangle(CanvasTriangle triangle, int textureFileIndex) {
  vector<CanvasTriangle> triangles = splitTriangle(triangle); // Preserves texture points (calculates new texturePoint for midpoint)
  vector<vector<uint32_t>> texturePixelValues = textureFiles[textureFileIndex]; // Loads texture from file
  
  for (int i = 0; i < 2; i++) { // Loops through both flat-bottomed triangles
    // Finds the number of steps to use for linear interpolation of canvas triangle
    float lineLeftNumberOfSteps = getLerpNumberOfSteps(triangles[i].vertices[0], triangles[i].vertices[1]);
    float lineRightNumberOfSteps = getLerpNumberOfSteps(triangles[i].vertices[0], triangles[i].vertices[2]);
    float numberOfSteps = std::max(lineLeftNumberOfSteps, lineRightNumberOfSteps);

    // Performs linear interpolation on canvas triangle
    vector<CanvasPoint> lineLeft  = interpolate2D(triangles[i].vertices[0], triangles[i].vertices[1], numberOfSteps);
    vector<CanvasPoint> lineRight = interpolate2D(triangles[i].vertices[0], triangles[i].vertices[2], numberOfSteps);

    // Finds the number of steps to use for linear interpolation of texture triangle
    float lineLeftNumberOfStepsTex = getLerpNumberOfSteps(triangles[i].vertices[0].texturePoint, triangles[i].vertices[1].texturePoint);
    float lineRightNumberOfStepsTex = getLerpNumberOfSteps(triangles[i].vertices[0].texturePoint, triangles[i].vertices[2].texturePoint);
    float numberOfStepsTex = std::max(lineLeftNumberOfStepsTex, lineRightNumberOfStepsTex);
    
    // Performs linear interpolation on texture triangle
    vector<CanvasPoint> lineLeftTex  = interpolate2D(triangles[i].vertices[0].texturePoint, triangles[i].vertices[1].texturePoint, numberOfStepsTex);
    vector<CanvasPoint> lineRightTex = interpolate2D(triangles[i].vertices[0].texturePoint, triangles[i].vertices[2].texturePoint, numberOfStepsTex);

    for (size_t j = 0; j < numberOfSteps; j++) { // Loops through canvas interpolation
      float ratio = j / (numberOfSteps-1); // Gets ratio to convert between canvas and texture space
      float textureIndex = round(ratio * (numberOfStepsTex-1)); // Uses ratio to get correct texture points on texture triangle
      float rakeLength = getLerpNumberOfSteps(lineLeft[j], lineRight[j]); // Calculates length between points on either side of canvas triangle to traverse across
      vector<CanvasPoint> canvasPositions = interpolate2D(lineLeft[j], lineRight[j], rakeLength);
      vector<CanvasPoint> texturePositions = interpolate2D(lineLeftTex[textureIndex], lineRightTex[textureIndex], rakeLength);

      for (int k = 0; k < rakeLength; k++) { // Draws line on between points on canvas triangle with texture points rather than colours
        if (isPixelOnScreen(canvasPositions[k].x, canvasPositions[k].y)) {
          window.setPixelColour(canvasPositions[k].x, canvasPositions[k].y, texturePixelValues[texturePositions[k].y][texturePositions[k].x]);
        }
      }
    }
  }
}

void threadRaytrace() {
    thread th1(drawFilledTrianglesRaytrace, 0, WIDTH/2, 0, HEIGHT/2);
    thread th2(drawFilledTrianglesRaytrace, WIDTH/2, WIDTH, 0, HEIGHT/2);
    thread th3(drawFilledTrianglesRaytrace, 0, WIDTH/2, HEIGHT/2, HEIGHT);
    thread th4(drawFilledTrianglesRaytrace, WIDTH/2, WIDTH, HEIGHT/2, HEIGHT);

    th1.join(); th2.join(); th3.join(); th4.join();
}

void drawFilledTrianglesRaytrace(int x0, int x1, int y0, int y1) {
  for (int x = x0; x < x1; x++) {
    for (int y = y0; y < y1; y++) {
      vec3 rayDirection = vec3(x - WIDTH/2, y - HEIGHT/2, FOCALLENGTH); 
      RayTriangleIntersection intersection = getClosestIntersection(rayDirection);
      int flippedX = WIDTH - x - 1;

      if (intersection.distanceFromCamera == INF) window.setPixelColour(flippedX, y, BLACK);
      else {
        Colour colour = intersection.colour;
        float brightness = calculateBrightness(intersection);
        
        uint32_t packedColour = (255<<24) + (int(colour.red*brightness)<<16) + (int(colour.green*brightness)<<8) + int(colour.blue*brightness);      
        window.setPixelColour(flippedX, y, packedColour);
      }     
    }
  }
}

RayTriangleIntersection getClosestIntersection(vec3 rayDirection) {
  vec3 closestIntersection = vec3(INF);
  RayTriangleIntersection closestInt = RayTriangleIntersection(vec3(INF), INF, ModelTriangle(), Colour(255,255,255));
  closestInt.intersectedTriangle.colour = Colour(0, 0, 0);

  for (int i = 0; i < (int)modelTriangles.size(); i++) {
    vec3 e0 = modelTriangles[i].vertices[1] - modelTriangles[i].vertices[0];
    vec3 e1 = modelTriangles[i].vertices[2] - modelTriangles[i].vertices[0];
    vec3 SPVector = camera.position - modelTriangles[i].vertices[0];
    mat3 DEMatrix(-rayDirection, e0, e1);
    vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;

    if (isInTriangle(possibleSolution) && abs(possibleSolution.x) <= abs(closestIntersection.x)) {
      closestIntersection = possibleSolution;
      vec3 position = camera.position + (closestIntersection.x * rayDirection);
      if (modelTriangles[i].textureFileIndex != -1) {
          closestInt = RayTriangleIntersection(position, closestIntersection.x, modelTriangles[i], getTextureIntersection(closestIntersection, modelTriangles[i]));
      }
      else closestInt = RayTriangleIntersection(position, closestIntersection.x, modelTriangles[i], modelTriangles[i].colour);
    }
  }

  return closestInt;
}

Colour getTextureIntersection(vec3 intersection, ModelTriangle modelTriangle) {
  vec2 e0 = modelTriangle.textureVertices[1] - modelTriangle.textureVertices[0];
  vec2 e1 = modelTriangle.textureVertices[2] - modelTriangle.textureVertices[0];
  vec2 value = modelTriangle.textureVertices[0] + (e0 * intersection.y) + (e1 * intersection.z);
  uint32_t colour = textureFiles[modelTriangle.textureFileIndex][value.x][value.y];
  
  return Colour((colour & 0x00FF0000) >> 16, (colour & 0x0000FF00) >> 8, (colour & 0x000000FF) >> 0);
}

float calculateBrightness(RayTriangleIntersection intersection) {
  vec3 pointLightVector = pointLight.position - intersection.intersectionPoint;
  float r = length(pointLightVector);
  float brightness = 0;

  vec3 e0 = intersection.intersectedTriangle.vertices[1] - intersection.intersectedTriangle.vertices[0];
  vec3 e1 = intersection.intersectedTriangle.vertices[2] - intersection.intersectedTriangle.vertices[0];

  if (!isInShadow(pointLightVector, r, intersection)) {
    brightness = pointLight.strength / (2 * PI * r * r);
    float normalisedAngleOfInc = dot(normalize(cross(e0, e1)), normalize(pointLightVector));
    if (normalisedAngleOfInc < 0) normalisedAngleOfInc = 0;
    brightness *= normalisedAngleOfInc;
  }

  if (brightness > 1.0) brightness = 1.0;
  else if ( brightness < ambientBrightness) brightness = ambientBrightness;

  return brightness;
}

bool isInShadow(vec3 pointLightVector, float distanceToLight, RayTriangleIntersection self) {
  for (int i = 0; i < (int)modelTriangles.size(); i++) {
    if (isTriangleSelf(self.intersectedTriangle, modelTriangles[i])) continue;
    vec3 e0 = modelTriangles[i].vertices[1] - modelTriangles[i].vertices[0];
    vec3 e1 = modelTriangles[i].vertices[2] - modelTriangles[i].vertices[0];
    vec3 SLVector = pointLight.position - modelTriangles[i].vertices[0];
    mat3 DEMatrix(-normalize(pointLightVector), e0, e1);
    vec3 possibleSolution = glm::inverse(DEMatrix) * SLVector;

    if (isInTriangle(possibleSolution) && abs(possibleSolution.x) < abs(distanceToLight) && possibleSolution.x < 0.001f) {
      return true;
    }
  }
  return false;
}

inline bool isInTriangle(vec3 intersectionPoint) {
  if (0.0 <= intersectionPoint.y && intersectionPoint.y <= 1.0 &&
      0.0 <= intersectionPoint.z && intersectionPoint.z <= 1.0 &&
      intersectionPoint.y + intersectionPoint.z <= 1.0 ) return true;
  else return false;
}

inline bool isTriangleSelf(ModelTriangle self, ModelTriangle tri) {
  if (self.vertices[0] != tri.vertices[0] || self.vertices[1] != tri.vertices[1] || self.vertices[2] != tri.vertices[2]) {
    return false;
  }
  else return true;
}

// 2D linear interpolation
vector<CanvasPoint> interpolate2D(CanvasPoint from, CanvasPoint to, float numberOfSteps) {
  float stepx = (to.x - from.x) / (numberOfSteps);
  float stepy = (to.y - from.y) / (numberOfSteps);
  std::vector<CanvasPoint> list2D;

  for(int i = 0; i < numberOfSteps; i++) {
    CanvasPoint point = CanvasPoint((from.x + (i * stepx)), (from.y + (i * stepy)));
    list2D.push_back(point);
  }

  return list2D;
}

vector<CanvasPoint> interpolate2D(TexturePoint from, TexturePoint to, float numberOfSteps) {
  float stepx = (to.x - from.x) / (numberOfSteps);
  float stepy = (to.y - from.y) / (numberOfSteps);
  std::vector<CanvasPoint> list2D;

  for(int i = 0; i < numberOfSteps; i++) {
    CanvasPoint point = CanvasPoint((from.x + (i * stepx)), (from.y + (i * stepy)));
    list2D.push_back(point);
  }

  return list2D;
}

// 3D linear interpolation
vector<CanvasPoint> interpolate3D(CanvasPoint from, CanvasPoint to, float numberOfSteps) {
  float stepx = (to.x - from.x) / (numberOfSteps);
  float stepy = (to.y - from.y) / (numberOfSteps);
  double stepz = (to.depth - from.depth) / (numberOfSteps);

  std::vector<CanvasPoint> list3D;
  for(int i = 0; i < numberOfSteps; i++) {
    CanvasPoint point = CanvasPoint(from.x + (i * stepx), from.y + (i * stepy), from.depth + (i * stepz));
    list3D.push_back(point);
  }

  return list3D;
}

// Obtains the correct number of steps for a linear interpolation
float getLerpNumberOfSteps (CanvasPoint from, CanvasPoint to) {
  float xDiff = to.x - from.x;
  float yDiff = to.y - from.y;

  return std::max(abs(xDiff), abs(yDiff));
}

float getLerpNumberOfSteps (TexturePoint from, TexturePoint to) {
  float xDiff = to.x - from.x;
  float yDiff = to.y - from.y;

  return std::max(abs(xDiff), abs(yDiff));
}

void drawStrokedTriangle(CanvasTriangle triangle) {
   drawLine(triangle.vertices[0], triangle.vertices[1], triangle.colour);
   drawLine(triangle.vertices[1], triangle.vertices[2], triangle.colour);
   drawLine(triangle.vertices[2], triangle.vertices[0], triangle.colour);
}

// Split triangle retruns one or two flat bottomed triangles with the highest
// vertex first
vector<CanvasTriangle> splitTriangle(CanvasTriangle triangle) {
    // Find the point with the highest Y value, the middle Y value and the lowest Y value
    vector<CanvasTriangle> splitTriangles;
    CanvasPoint maxPoint, midPoint, minPoint;

    float maxY = std::max(std::max(triangle.vertices[0].y, triangle.vertices[1].y), triangle.vertices[2].y);
    float minY = std::min(std::min(triangle.vertices[0].y, triangle.vertices[1].y), triangle.vertices[2].y);

    // Find max and min y values on points, accounting for when there are two maxes/mins
    bool maxAssigned, minAssigned = false;
    for (int i = 0; i < 3; i++) {
      if(maxY == triangle.vertices[i].y && !maxAssigned) {
        maxPoint = triangle.vertices[i];
        maxAssigned = true;
      }
      else if(minY == triangle.vertices[i].y && !minAssigned) {
        minPoint = triangle.vertices[i];
        minAssigned = true;
      }
      else midPoint = triangle.vertices[i];
    }
   
    // Check if triangle is already flatBottomed, if it is then there is no need
    // to split it so just repack it with the vertices in the correct order
    if (isFlatBotttomedTriangle(triangle)) {
      splitTriangles.push_back(CanvasTriangle(maxPoint, midPoint, minPoint, triangle.colour));
      CanvasPoint nullPoint = CanvasPoint(0,0);
      CanvasTriangle bottomTriangle = CanvasTriangle(nullPoint, nullPoint, nullPoint, triangle.colour);
      splitTriangles.push_back(bottomTriangle);

      return splitTriangles;
    }
    
    CanvasPoint interpolatedPoint = getInterpolatedPoint(maxPoint, midPoint, minPoint);
    
    // Define both flat bottomed triangles
    CanvasTriangle topTriangle = CanvasTriangle(maxPoint, midPoint, interpolatedPoint, triangle.colour);
    CanvasTriangle bottomTriangle = CanvasTriangle(minPoint, midPoint, interpolatedPoint, triangle.colour);
    splitTriangles.push_back(topTriangle);
    splitTriangles.push_back(bottomTriangle);
    
    return splitTriangles;
}

// Interpolate down the line from the maxPoint to the minPoint until you
// reach the Y value of the middle point, this point can then be used
// to define both flat bottomed triangles
CanvasPoint getInterpolatedPoint(CanvasPoint maxPoint, CanvasPoint midPoint, CanvasPoint minPoint) {
  float numberOfSteps = getLerpNumberOfSteps(maxPoint, minPoint);
  vector<CanvasPoint> points = interpolate3D(maxPoint, minPoint, numberOfSteps);

  CanvasPoint interpolatedPoint;
  for (float i = 0.0; i < numberOfSteps; i++) {
    if (round(points[i].y) == round(midPoint.y)) { // Found the correct y value
      interpolatedPoint = CanvasPoint(points[i].x, points[i].y, points[i].depth); // Create the correct point
      
      if (maxPoint.texturePoint.x != -1) { // Find the interpolated texture point if texture points are set
        float numberOfStepsTex = getLerpNumberOfSteps(maxPoint.texturePoint, minPoint.texturePoint);
        if (numberOfStepsTex == 0) numberOfStepsTex = 1;
        vector<CanvasPoint> textureLine = interpolate2D(maxPoint.texturePoint, minPoint.texturePoint, numberOfStepsTex);
        //cout << textureLine.size() << " " << i << " " << i/numberOfSteps << " " << numberOfSteps << " " << numberOfStepsTex << endl;
        CanvasPoint interpolatedPointTex = textureLine[round((i / numberOfSteps) * numberOfStepsTex) ];
        interpolatedPoint.texturePoint = TexturePoint(round(interpolatedPointTex.x), round(interpolatedPointTex.y));       
      }
    }
  }

  return interpolatedPoint;
}

inline bool isPixelOnScreen(int x, int y) {
  if (y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH) return false;
  else return true;
}

bool isFlatBotttomedTriangle(CanvasTriangle triangle) {
  float y0 = round(triangle.vertices[0].y);
  float y1 = round(triangle.vertices[1].y);
  float y2 = round(triangle.vertices[2].y);

  if (y0 == y1 || y0 == y2 || y1 == y2 ) return true;
  else return false;
}

void handleEvent(SDL_Event event) {
  if(event.type == SDL_KEYDOWN) {
    if(event.key.keysym.sym == SDLK_LEFT) {
      cout << "LEFT" << endl;
      camera.translate(-1, 0, 0);
      update();
    }
    else if(event.key.keysym.sym == SDLK_RIGHT) {
      cout << "RIGHT" << endl;
      camera.translate(1, 0, 0);
      update();
    }
    else if(event.key.keysym.sym == SDLK_UP) {
      cout << "UP" << endl;
      camera.translate(0, 1, 0);
      update();
    }
    else if(event.key.keysym.sym == SDLK_DOWN) {
      cout << "DOWN" << endl;
      camera.translate(0, -1, 0);
      update();
    }
    else if(event.key.keysym.sym == SDLK_f) {
      cout << "F" << endl;
      camera.translate(0, 0, -1);
      update();
    }
    else if(event.key.keysym.sym == SDLK_b) {
      cout << "B" << endl;
      camera.translate(0, 0, 1);
      update();
    }
    else if(event.key.keysym.sym == SDLK_w) {
      cout << "W" << endl;
      camera.rotateX(5);
      update();
    }
    else if(event.key.keysym.sym == SDLK_s) {
      cout << "S" << endl;
      camera.rotateX(-5);
      update();
    }
    else if(event.key.keysym.sym == SDLK_a) {
      cout << "A" << endl;
      camera.rotateY(5);
      update();
    }
    else if(event.key.keysym.sym == SDLK_d) {
      cout << "D" << endl;
      camera.rotateY(-5);
      update();
    }
    else if(event.key.keysym.sym == SDLK_1) {
      cout << "WIREFRAME" << endl;
      camera.drawingMode = WIREFRAME;
      update();
    }
    else if(event.key.keysym.sym == SDLK_2) {
      cout << "RASTERISE" << endl;
      camera.drawingMode = RASTERISE;
      update();
    }
    else if(event.key.keysym.sym == SDLK_3) {
      cout << "RAYTRACE" << endl;
      camera.drawingMode = RAYTRACE;
      update();
    }
    else if(event.key.keysym.sym == SDLK_p) {
      cout << "SAVINGPPM" << endl;
      savePPM();
    }
  }
  else if(event.type == SDL_MOUSEBUTTONDOWN) std::cout << "MOUSE CLICKED" << std::endl;
}

// Loads the colour palette from OBJ material file into hash map
unordered_map<string, Colour> loadMaterial(string file) {
  ifstream ifs;
  ifs.open (file, std::ifstream::in);

  unordered_map<string, Colour> colourPalette;

  char newMtl[256];
  ifs.getline(newMtl, 256);
  string newMtlString = newMtl;
  size_t tmp = newMtlString.find("newmtl");

  while(tmp != string::npos) {
    string name = newMtlString.substr(7, newMtlString.length() - 1);

    char value[256];
    ifs.getline(value, 256);
    string valueString = value;

    string* tokens = split(valueString, ' ');
    Colour colourValue = Colour(stof(*(tokens + 1))*255, stof(*(tokens + 2))*255, stof(*(tokens + 3))*255);

    char emptyLine[256];
    ifs.getline(emptyLine, 256);
    ifs.getline(newMtl, 256);
    newMtlString = newMtl;
    tmp = newMtlString.find("newmtl");
    colourPalette.insert({name, colourValue});
  }
  
  return colourPalette;
}

// Loads all triangle geometry from OBJ file into ModelTrangles vector
void loadOBJ(const char* OBJFile, float scaleFactor) { 
  unordered_map<string, Colour> colourPalette;
  vector<vec3> vertices;
  vector<vec2> textureVertices;
  vector<vector<uint32_t>> ppmFile;
  string materialFile;
  
  ifstream ifs;
  ifs.open (OBJFile, std::ifstream::in);

  while( ifs.peek() != EOF || ifs.peek() != -1 )
  {
    Colour currentColour;
    char line[256];
    ifs.getline(line, 256);
    string lineString = line;

    if (lineString.find("mtllib") != string::npos){
      string* mtlTokens = split(lineString, ' ');
      materialFile = *(mtlTokens + 1);
      materialFile.erase(remove_if(materialFile.begin(), materialFile.end(), ::isspace), materialFile.end());
      
      ifstream mtlIfs;
      mtlIfs.open(materialFile, std::ifstream::in);
      char mtlLine[256];
      mtlIfs.getline(mtlLine, 256);
      lineString = mtlLine;

      if (lineString.find("map_Kd") != string::npos) {
        string* tokens = split(lineString, ' ');
        ppmFile = loadPPM(*(tokens + 1));
        textureFiles.push_back(ppmFile);
      }
      else colourPalette = loadMaterial(materialFile);
    }
        
    if (lineString.find("usemtl") != string::npos) {
      string* tokens = split(lineString, ' ');
      currentColour = colourPalette.at(*(tokens + 1));  
    }
    else if (line[0] == 'v' && line[1] == 't') {
      string* tokens = split(lineString, ' ');
      textureVertices.push_back(vec2(stof(*(tokens + 1)), stof(*(tokens + 2))));
    }
    else if (line[0] == 'v') {
      string* tokens = split(lineString, ' ');
      vertices.push_back(vec3(stof(*(tokens + 1)), stof(*(tokens + 2)), stof(*(tokens + 3))));
    }
    else if (line[0] == 'f') {
      string* spaceTokens = split(lineString, ' ');
      vector<string> v;
      vector<string> vt;
      for (int i = 1; i < 4; i++) {
        string* slashTokens = split(*(spaceTokens + i), '/');
        v.push_back(*(slashTokens + 0));
        if (!textureVertices.empty()) vt.push_back(*(slashTokens + 1));
      }
      if (textureVertices.empty()) modelTriangles.push_back(ModelTriangle( vertices[stoi(v[0]) - 1]*scaleFactor,  vertices[stoi(v[1]) - 1]*scaleFactor, vertices[stoi(v[2]) - 1]*scaleFactor, currentColour));
      else {
        int ppmWidth = (int)ppmFile[1].size()-1;
        int ppmHeight = (int)ppmFile[0].size()-1;
        
        vec2 texturePoint0 = vec2(round(textureVertices[stoi(vt[0]) - 1].x * ppmWidth), round(textureVertices[stoi(vt[0]) - 1].y * ppmHeight));
        vec2 texturePoint1 = vec2(round(textureVertices[stoi(vt[1]) - 1].x * ppmWidth), round(textureVertices[stoi(vt[1]) - 1].y * ppmHeight));
        vec2 texturePoint2 = vec2(round(textureVertices[stoi(vt[2]) - 1].x * ppmWidth), round(textureVertices[stoi(vt[2]) - 1].y * ppmHeight));
        modelTriangles.push_back(ModelTriangle( vertices[stoi(v[0]) - 1]*scaleFactor,  vertices[stoi(v[1]) - 1]*scaleFactor, vertices[stoi(v[2]) - 1]*scaleFactor, texturePoint0, texturePoint1, texturePoint2, textureFiles.size()-1)); 
      }
    }
  } 
}

// Loads a texture from a ppm into a 2D vector
vector<vector<uint32_t>> loadPPM(string file) {
  ifstream ifs;
  ifs.open (file, std::ifstream::in);

  char format[256], comment[256], widthAndHeight[256], colourChannel[256];
  ifs.getline(format, 256);
  ifs.getline(comment, 256);
  ifs.getline(widthAndHeight, 256);
  ifs.getline(colourChannel, 256);

  string s = widthAndHeight;

  size_t found = s.find(" ");
  string widthStr = s.substr(0, found);
  string heightStr = s.substr(found+1, s.length()-1);

  int width = stoi(widthStr, nullptr, 10);
  int height = stoi(heightStr, nullptr, 10);

  vector<vector<uint32_t>> pixelValues;

  for(int y=0; y < height; y++) {
    vector<uint32_t> colours;

    for(int x=0; x < width; x++) {
      uint32_t colour = (255<<24) + (int(ifs.get())<<16) + (int(ifs.get())<<8) + int(ifs.get());
      colours.push_back(colour);
    }
    pixelValues.push_back(colours);
  }

  return pixelValues;
}

void savePPM() {
  ofstream PPMFile;
  string name = "file" + to_string(savedPPMs) + ".ppm";
  PPMFile.open(name);
  PPMFile << "P6\n";
  PPMFile << "#Created by Jangus Parsonkeen\n";
  PPMFile << to_string(WIDTH) + " " + to_string(HEIGHT) + "\n";
  PPMFile << "255\n";

  for (int y=0; y < HEIGHT; y++) {
    for (int x=0; x<WIDTH; x++) {
      char r = window.getPixelColour(x,y) >> 16;
      char g = window.getPixelColour(x,y) >> 8;
      char b = window.getPixelColour(x,y);
      PPMFile << r << g << b;
    }
  }

  PPMFile.close();
  savedPPMs++;
}
