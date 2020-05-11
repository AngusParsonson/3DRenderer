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
#include <time.h>
#include <boost/thread/thread.hpp>

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

struct AreaLight {
  vector<PointLight> pointLights;
  AreaLight(vec3 pos, float str, float spread, int numLights) {
    srand(time(0));
    pointLights.push_back(PointLight(pos, str));

    for (int i = 0; i < numLights; i++) {
      float x; float y; float z;

      int xRan = rand()%2;
      if (xRan == 0)  x = pos.x + (spread * 1.0f/(rand()%100));
      else x = pos.x - (spread * 1.0f/(rand()%100));

      int yRan = rand()%2;
      if (yRan == 0)  y = pos.y + (spread * 1.0f/(rand()%100));
      else y = pos.y - (spread * 1.0f/(rand()%100));

      int zRan = rand()%2;
      if (zRan == 0)  z = pos.z + (spread * 1.0f/(rand()%100));
      else z = pos.z - (spread * 1.0f/(rand()%100));

      pointLights.push_back(PointLight(vec3(x, y, z), str));
    }
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
bool handleObjectEvent(SDL_Event event, string objectName);
CanvasTriangle get2DProjection(ModelTriangle modelTriangle, Camera camera);
void drawLine(CanvasPoint from, CanvasPoint to, Colour colour);

//Wu Line Optimisation
void drawWuAALine(CanvasPoint from, CanvasPoint to, Colour colour);
void drawPixel(int x, int y, Colour colour);
int iPartOfNumber(float x);
float fPartOfNumber(float x);
float rfPartOfNumber(float x);
void WuStrokedTriangle(CanvasTriangle triangle);

//
void drawStrokedTriangle(CanvasTriangle triangle);
void drawFilledTriangle(CanvasTriangle triangle);
void drawTexturedTriangle(CanvasTriangle triangle, int textureFileIndex);
void drawFilledTrianglesRaytrace(int x0, int x1, int y0, int y1);
uint32_t getRaytraceColour(int x, int y, float offsetX, float offsetY, vector<ModelTriangle> unculledTriangles);
void threadRaytrace(int numThreads);
RayTriangleIntersection getClosestIntersection(vec3 rayDirection, vec3 fromPosition, vector<ModelTriangle> unculledTriangles);
RayTriangleIntersection getClosestIntersection(vec3 rayDirection, vec3 fromPosition, ModelTriangle self, vector<ModelTriangle> unculledTriangles);
Colour getTextureIntersection(vec3 intersection, ModelTriangle modelTriangle);
bool isInTriangle(vec3 intersectionPoint);
bool isTriangleSelf(ModelTriangle self, ModelTriangle tri);
float calculateBrightness(RayTriangleIntersection intersection);
bool isInShadow(vec3 pointLightVector, float distanceToLight, RayTriangleIntersection self, float pointLight);
float softShadow(RayTriangleIntersection self);
float phongShader(RayTriangleIntersection intersection);
Colour mirror(RayTriangleIntersection intersection);
Colour glass(RayTriangleIntersection intersection, vec3 fromPosition, vector<ModelTriangle> unculledTriangles);
float getFresnel(vec3 incidentRay, vec3 pixelNormal, float refractiveIndex);
vec3 getRefraction(vec3 incidentRay, vec3 pixelNormal, float refractiveIndex);
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
vector<ModelTriangle> backFaceCull();
bool pixelClipping(float z);
inline bool isMirror(ModelTriangle triangle);
inline bool isGlass(ModelTriangle triangle);
inline vec3 getMiddle(ModelTriangle triangle);
void generateObjectMap();

uint32_t BLACK = (255<<24) + (int(0)<<16) + (int(0)<<8) + int(0);
double depthBuffer[WIDTH][HEIGHT] = { { INF } };
int savedPPMs = 0;
unordered_map<string, vector<int>> objectMap;

DrawingWindow window = DrawingWindow(WIDTH, HEIGHT, false);
Camera camera = Camera(0, 2, 4);
//PointLight pointLight = PointLight(vec3(-0.3, 4.8, -3.1), 150);
AreaLight areaLight = AreaLight(vec3(-0.3, 3.5, -3.1), 150, 2, 0);
float ambientBrightness = areaLight.pointLights[0].strength / 350;
float specularExponent = 16;
float diffuse = 0.5;

vector<ModelTriangle> modelTriangles;
vector<CanvasTriangle> canvasTriangles;
vector<vector<vector<uint32_t>>> textureFiles;

int main(int argc, char* argv[])
{
  SDL_Event event;
  //loadOBJ("cornell-box.obj", 1);
  loadOBJ("logo.obj", 0.01);
  loadOBJ("portal.obj", 1);
  //loadOBJ("discoBall.obj", 1);
  //loadOBJ("sphere.obj", 0.5);
  generateObjectMap();
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
  vector<ModelTriangle> unculledTriangles = backFaceCull();
  for (int i = 0; i < (int)unculledTriangles.size(); i++) {
    newCanvasTriangles.push_back(get2DProjection(unculledTriangles[i], camera));
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

vector<ModelTriangle> backFaceCull() {
  vector<ModelTriangle> unculledTriangles;
  for (int i = 0; i < (int)modelTriangles.size(); i++) {
    vec3 e0 = modelTriangles[i].vertices[1] - modelTriangles[i].vertices[0];
    vec3 e1 = modelTriangles[i].vertices[2] - modelTriangles[i].vertices[0];
    vec3 middle = (modelTriangles[i].vertices[0] + modelTriangles[i].vertices[1] + modelTriangles[i].vertices[2]) / 3.0f;
    vec3 camPos = camera.position * camera.rotation;
    vec3 diff = normalize(camPos - middle);

    vec3 faceDir = normalize(cross(e1, e0));
    float dotProd = dot(diff, faceDir);

    if (dotProd < 0) unculledTriangles.push_back(modelTriangles[i]);
  }

  return unculledTriangles;
}

bool pixelClipping(float z) {
  vec3 camPos = camera.position * camera.rotation;
  float diff = camPos.z - 1/z;
  if (diff <= 0) return false;
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
      //drawStrokedTriangle(canvasTriangles[i]);
      WuStrokedTriangle(canvasTriangles[i]);
    }
  }

  else if (camera.drawingMode == RAYTRACE) threadRaytrace(64);
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

// LINE OPTIMISATION

int iPartOfNumber(float x)
{
    return floor(x);
}


//returns fractional part of a number
float fPartOfNumber(float x)
{
    if (x>0) return x - iPartOfNumber(x);
    else return x - (iPartOfNumber(x)+1);

}

//returns 1 - fractional part of number
float rfPartOfNumber(float x)
{
    return 1 - fPartOfNumber(x);
}

void drawPixel(int x, int y, Colour colour, double z) {
  uint32_t packedColour = (255<<24) + (int(colour.red)<<16) + (int(colour.green)<<8) + int(colour.blue);
  if (isPixelOnScreen(x, y)) { // Pixel is within resolution
    if(depthBuffer[(int)x][(int)y] > z) { // Pixel is closer to camera than previous pixel
        depthBuffer[(int)x][(int)y] = z;
        window.setPixelColour(x, y, packedColour);
    }
  }
}

void drawWuAALine(CanvasPoint from, CanvasPoint to, Colour colour){
  float x0 = from.x; float x1 = to.x;
  float y0 = from.y; float y1 = to.y;
  double z0 = from.depth; double z1 = to.depth;

  int steep = abs(y1 - y0) > abs(x1 - x0);

  if (steep) {
      swap(x0 , y0);
      swap(x1 , y1);
  }
  if (x0 > x1){
      swap(x0 ,x1);
      swap(y0 ,y1);
      swap(z0, z1);
  }

  //compute the slope
  float dx = x1-x0;
  float dy = y1-y0;
  float dz = z1-z0;
  float gradient = dy/dx;
  float zGradient = dz/dx;
  if (dx == 0.0)
      gradient = 1;

  int xpxl1 = x0;
  int xpxl2 = x1;
  float intersectY = y0;
  double intersectZ = z0;

  // main loop
  if (steep)
  {
      int x;
      for (x = xpxl1 ; x <=xpxl2 ; x++)
      {
          // pixel coverage is determined by fractional
          // part of y co-ordinate
          drawPixel(iPartOfNumber(intersectY), x,
                      colour, intersectZ);
          drawPixel(iPartOfNumber(intersectY)-1, x,
                      colour, intersectZ);
          intersectY += gradient;
          intersectZ += zGradient;
      }
  }
  else
  {
      int x;
      for (x = xpxl1 ; x <=xpxl2 ; x++)
      {
          // pixel coverage is determined by fractional
          // part of y co-ordinate
          drawPixel(x, iPartOfNumber(intersectY),
                      colour, intersectZ);
          drawPixel(x, iPartOfNumber(intersectY)-1,
                        colour, intersectZ);
          intersectY += gradient;
          intersectZ += zGradient;
      }
  }

}

void WuStrokedTriangle(CanvasTriangle triangle) {
  drawWuAALine(triangle.vertices[0], triangle.vertices[1], triangle.colour);
  drawWuAALine(triangle.vertices[0], triangle.vertices[2], triangle.colour);
  drawWuAALine(triangle.vertices[1], triangle.vertices[2], triangle.colour);
}

// Triangle Rasteriser
// Works by splitting into two flat bottomed tirangles, and then drawing
// lines horizontally across both triangle
void drawFilledTriangle(CanvasTriangle triangle) {
  vector<CanvasTriangle> triangles = splitTriangle(triangle);

  for (int i = 0; i < 2; i++) {
    //drawStrokedTriangle(triangles[i]);
    WuStrokedTriangle(triangles[i]);
    float lineLeftNumberOfSteps = getLerpNumberOfSteps(triangles[i].vertices[0], triangles[i].vertices[1]);
    float lineRightNumberOfSteps = getLerpNumberOfSteps(triangles[i].vertices[0], triangles[i].vertices[2]);
    float numberOfSteps = std::max(lineLeftNumberOfSteps, lineRightNumberOfSteps);

    vector<CanvasPoint> lineLeft  = interpolate3D(triangles[i].vertices[0], triangles[i].vertices[1], numberOfSteps);
    vector<CanvasPoint> lineRight = interpolate3D(triangles[i].vertices[0], triangles[i].vertices[2], numberOfSteps);

    for (size_t j = 0; j < numberOfSteps; j++) {
      //drawLine(lineLeft[j], lineRight[j], triangle.colour);
      drawWuAALine(lineLeft[j], lineRight[j], triangle.colour);
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

void threadRaytrace(int numThreads) {
  vector<boost::thread*> threads;
  int rowColLength = sqrt(numThreads);
  int xSize = WIDTH / rowColLength;
  int ySize = HEIGHT / rowColLength;

  for (int x = 0; x < rowColLength; x++) {
    int xStart = x*xSize;
    int xEnd = xStart + xSize;

    for (int y = 0; y < rowColLength; y++) {
      int yStart = y*ySize;
      int yEnd = yStart + ySize;
      boost::thread* name = new boost::thread(drawFilledTrianglesRaytrace, xStart, xEnd, yStart, yEnd);
      threads.push_back(name);
    }
  }
  for (int i = 0; i < (int)threads.size(); i++) threads[i]->join();


  //drawFilledTrianglesRaytrace(0, WIDTH, 0, HEIGHT);
}

void drawFilledTrianglesRaytrace(int x0, int x1, int y0, int y1) {
  vector<ModelTriangle> unculledTriangles = backFaceCull();
  int w = x1 - x0 + 1; int h = y1 - y0 + 1;
  uint32_t** offsetPixels = new uint32_t*[w];
  for(int i = 0; i < w; ++i) {
    offsetPixels[i] = new uint32_t[h];
    for (int j = 0; j < h; j++) offsetPixels[i][j] = 0;
  }

  for (int x = x0; x < x1; x++) {
    for (int y = y0; y < y1; y++) {
      uint32_t pixel = getRaytraceColour(x, y, 0, 0, unculledTriangles);
      int oX = x % (x1 - x0); int oY = y % (y1 - y0);
      if (offsetPixels[oX][oY] == 0) offsetPixels[oX][oY] = getRaytraceColour(x, y, -0.5, -0.5, unculledTriangles);
      if (offsetPixels[oX+1][oY] == 0) offsetPixels[oX+1][oY] = getRaytraceColour(x, y, 0.5, 0.5, unculledTriangles);
      if (offsetPixels[oX][oY+1] == 0) offsetPixels[oX][oY+1] = getRaytraceColour(x, y, -0.5, 0.5, unculledTriangles);
      if (offsetPixels[oX+1][oY+1] == 0) offsetPixels[oX+1][oY+1] = getRaytraceColour(x, y, 0.5, -0.5, unculledTriangles);
      uint32_t topLeft = offsetPixels[oX][oY]; uint32_t topRight = offsetPixels[oX+1][oY];
      uint32_t bottomLeft = offsetPixels[oX][oY+1]; uint32_t bottomRight = offsetPixels[oX+1][oY+1];

      int flippedX = WIDTH - x - 1;

      Colour colour;
      colour.red = (((pixel & 0x00FF0000) >> 16) * 0.5) + (((topLeft & 0x00FF0000) >> 16) + ((topRight & 0x00FF0000) >> 16) + ((bottomLeft & 0x00FF0000) >> 16) + ((bottomRight & 0x00FF0000) >> 16)) * 0.125;
      colour.green = (((pixel & 0x0000FF00) >> 8) * 0.5) + (((topLeft & 0x0000FF00) >> 8) + ((topRight & 0x0000FF00) >> 8) + ((bottomLeft & 0x0000FF00) >> 8) + ((bottomRight & 0x0000FF00) >> 8)) * 0.125;
      colour.blue = (((pixel & 0x000000FF) >> 0) * 0.5) + ((topLeft & 0x000000FF) + (topRight & 0x000000FF) + (bottomLeft & 0x000000FF) + (bottomRight & 0x000000FF)) * 0.125;

      uint32_t packedColour = (255<<24) + (int(colour.red)<<16) + (int(colour.green)<<8) + int(colour.blue);
      window.setPixelColour(flippedX, y, packedColour);
    }
  }
}

uint32_t getRaytraceColour(int x, int y, float offsetX, float offsetY, vector<ModelTriangle> unculledTriangles) {
  vec3 rayDirection = vec3(x - WIDTH/2 + offsetX, y - HEIGHT/2 + offsetY, FOCALLENGTH) * camera.rotation;
  RayTriangleIntersection intersection = getClosestIntersection(rayDirection, camera.position, unculledTriangles);

  if (intersection.localIntersection.x == INF) return BLACK;
  else {
    Colour colour;
    float brightness;
    uint32_t packedColour;
    if (intersection.intersectedTriangle.textureFileIndex != -1) {
      colour = getTextureIntersection(intersection.localIntersection, intersection.intersectedTriangle);
    }

    else if(isMirror(intersection.intersectedTriangle)){
      colour = mirror(intersection);
    }

    else if(isGlass(intersection.intersectedTriangle)) {
      colour = glass(intersection, camera.position, unculledTriangles);
    }

    else {
      colour = intersection.intersectedTriangle.colour;
    }
    if (!isMirror(intersection.intersectedTriangle))  {
      brightness = calculateBrightness(intersection);
      packedColour = (255<<24) + (int(colour.red*brightness)<<16) + (int(colour.green*brightness)<<8) + int(colour.blue*brightness);
    }
    else packedColour = (255<<24) + (int(colour.red)<<16) + (int(colour.green)<<8) + int(colour.blue);

    return packedColour;
  }
}

RayTriangleIntersection getClosestIntersection(vec3 rayDirection, vec3 fromPosition, vector<ModelTriangle> unculledTriangles) {
  vec3 closestIntersection = vec3(INF);
  RayTriangleIntersection closestInt = RayTriangleIntersection(vec3(INF), vec3(INF), ModelTriangle());
  closestInt.intersectedTriangle.colour = Colour(0, 0, 0);

  for (int i = 0; i < (int)unculledTriangles.size(); i++) {
    vec3 e0 = unculledTriangles[i].vertices[1] - unculledTriangles[i].vertices[0];
    vec3 e1 = unculledTriangles[i].vertices[2] - unculledTriangles[i].vertices[0];
    vec3 SPVector = fromPosition - unculledTriangles[i].vertices[0];
    mat3 DEMatrix(-rayDirection, e0, e1);
    vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;

    if (isInTriangle(possibleSolution) && abs(possibleSolution.x) <= abs(closestIntersection.x)) {
      closestIntersection = possibleSolution;
      vec3 position = fromPosition + (closestIntersection.x * rayDirection);
      closestInt = RayTriangleIntersection(position, closestIntersection, unculledTriangles[i]);
    }
  }
  return closestInt;
}

RayTriangleIntersection getClosestIntersection(vec3 rayDirection, vec3 fromPosition, ModelTriangle self, vector<ModelTriangle> unculledTriangles) {
  vec3 closestIntersection = vec3(INF);
  RayTriangleIntersection closestInt = RayTriangleIntersection(vec3(INF), vec3(INF), ModelTriangle("name", vec3(0),vec3(0),vec3(0),Colour(0,0,0)));
  //cout << fromPosition.x << " " << rayDirection.x << endl;
  for (int i = 0; i < (int)unculledTriangles.size(); i++) {
    vec3 e0 = unculledTriangles[i].vertices[1] - unculledTriangles[i].vertices[0];
    vec3 e1 = unculledTriangles[i].vertices[2] - unculledTriangles[i].vertices[0];
    vec3 SPVector = fromPosition - unculledTriangles[i].vertices[0];
    mat3 DEMatrix(-rayDirection, e0, e1);
    vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;

    if (isInTriangle(possibleSolution) && abs(possibleSolution.x) <= abs(closestIntersection.x) && !isTriangleSelf(self, unculledTriangles[i]) && !isMirror(unculledTriangles[i]) && !isGlass(unculledTriangles[i])) {
      closestIntersection = possibleSolution;
      vec3 position = fromPosition + (closestIntersection.x * rayDirection);
      closestInt = RayTriangleIntersection(position, closestIntersection, unculledTriangles[i]);
    }
  }
  //cout << closestInt.intersectionPoint.x << endl;
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
  vec3 pointLightVector = areaLight.pointLights[0].position - intersection.intersectionPoint;
  float r = length(pointLightVector);
  float brightness = 0;

  vec3 e0 = intersection.intersectedTriangle.vertices[1] - intersection.intersectedTriangle.vertices[0];
  vec3 e1 = intersection.intersectedTriangle.vertices[2] - intersection.intersectedTriangle.vertices[0];
    brightness = areaLight.pointLights[0].strength / (2 * PI * r * r);

  if (!intersection.intersectedTriangle.hasNormals) {
    float normalisedAngleOfInc = dot(normalize(cross(e0, e1)), normalize(pointLightVector));
    if (normalisedAngleOfInc < 0) normalisedAngleOfInc = 0;
    brightness *= normalisedAngleOfInc;
  }
  else brightness *= phongShader(intersection);

  if (!isGlass(intersection.intersectedTriangle)) brightness *= softShadow(intersection);

  if (brightness > 1.0) brightness = 1.0;
  else if ( brightness < ambientBrightness) brightness = ambientBrightness;

  return brightness;
}

float phongShader(RayTriangleIntersection intersection) {
  ModelTriangle intTriangle = intersection.intersectedTriangle;
  vec3 pointLightVector = normalize(areaLight.pointLights[0].position - intersection.intersectionPoint);
  vec3 cameraVector = normalize(camera.position - intersection.intersectionPoint);
  vec3 localInt = intersection.localIntersection;
  vec3 pixelNormal = normalize(((1 - localInt.y - localInt.z) * intTriangle.vertices[0]) + (localInt.y * intTriangle.vertices[1]) + (localInt.z * intTriangle.vertices[2]));
  vec3 reflection = normalize((2*dot(pointLightVector, pixelNormal) * pixelNormal) - pointLightVector);

  float result = pow(dot(cameraVector, reflection), specularExponent);
  return result;
}

Colour mirror(RayTriangleIntersection intersection) {
  ModelTriangle intTriangle = intersection.intersectedTriangle;
  vec3 cameraVector = normalize(camera.position - intersection.intersectionPoint);
  vec3 localInt = intersection.localIntersection;
  vec3 pixelNormal;
  Colour colour;
  if (intTriangle.hasNormals) {
    pixelNormal = normalize(((1 - localInt.y - localInt.z) * intTriangle.vertices[0]) + (localInt.y * intTriangle.vertices[1]) + (localInt.z * intTriangle.vertices[2]));
  }
  else {
    vec3 e0 = intTriangle.vertices[1] - intTriangle.vertices[0];
    vec3 e1 = intTriangle.vertices[2] - intTriangle.vertices[0];
    pixelNormal = normalize(cross(e1, e0));
  }

  vec3 reflection = normalize(cameraVector - (2*dot(cameraVector, pixelNormal) * pixelNormal));
  RayTriangleIntersection reflectedIntersection = getClosestIntersection(reflection, intersection.intersectionPoint, intTriangle, modelTriangles);
  if (reflectedIntersection.intersectedTriangle.textureFileIndex != -1) {
    colour = getTextureIntersection(intersection.localIntersection, intersection.intersectedTriangle);
  }
  else colour = reflectedIntersection.intersectedTriangle.colour;
  float brightness = calculateBrightness(reflectedIntersection);
  colour.red *= brightness;
  colour.blue *= brightness;
  colour.green *= brightness;

  return colour;
}

Colour glass(RayTriangleIntersection intersection, vec3 fromPosition, vector<ModelTriangle> unculledTriangles) {
  ModelTriangle intTriangle = intersection.intersectedTriangle;
  vec3 incidentRay = normalize(fromPosition - intersection.intersectionPoint);
  vec3 localInt = intersection.localIntersection;
  vec3 pixelNormal;
  if (intTriangle.hasNormals) {
    pixelNormal = normalize(((1 - localInt.y - localInt.z) * intTriangle.vertices[0]) + (localInt.y * intTriangle.vertices[1]) + (localInt.z * intTriangle.vertices[2]));
  }
  else {
    vec3 e0 = intTriangle.vertices[1] - intTriangle.vertices[0];
    vec3 e1 = intTriangle.vertices[2] - intTriangle.vertices[0];
    pixelNormal = normalize(cross(e1, e0));
  }
  vec3 refractedRay = getRefraction(incidentRay, pixelNormal, 1.5);
  float fresnal = getFresnel(incidentRay, pixelNormal, 1.5);
  Colour reflectedColour = mirror(intersection);

  RayTriangleIntersection refractedInt = getClosestIntersection(refractedRay, intersection.intersectionPoint, intTriangle, unculledTriangles);

  Colour colour = refractedInt.intersectedTriangle.colour;
  if (isGlass(refractedInt.intersectedTriangle)) {
    colour = glass(refractedInt, intersection.intersectionPoint, unculledTriangles);
  }

  colour.red = colour.red * (1-fresnal) + reflectedColour.red * fresnal;
  colour.blue = colour.blue * (1-fresnal) + reflectedColour.blue * fresnal;
  colour.green = colour.green * (1-fresnal) + reflectedColour.green * fresnal;
  return colour;
}

vec3 getRefraction(vec3 incidentRay, vec3 pixelNormal, float refractiveIndex) {
  float cosI = clamp(-1.0f, 1.0f, dot(incidentRay, pixelNormal)); // positive if leaving, negative if entering
  float etai = 1;
  vec3 n = pixelNormal;

  if (cosI < 0) cosI = -cosI;
  else {
    swap(etai, refractiveIndex);
    n = -pixelNormal;
  }
  vec3 reflection = incidentRay - (2*dot(incidentRay, pixelNormal) * pixelNormal);
  float eta = etai / refractiveIndex;
  float k = 1 - eta * eta * (1 - cosI * cosI);
  return k < 0 ? reflection : eta * incidentRay + (eta * cosI - sqrtf(k)) * n;
}

float getFresnel(vec3 incidentRay, vec3 pixelNormal, float refractiveIndex) {
  float cosi = clamp(-1.0f, 1.0f, dot(incidentRay, pixelNormal));
  float etai = 1; float kr;
  if (cosi > 0) { std::swap(etai, refractiveIndex); }
  // Compute sini using Snell's law
  float sint = etai / refractiveIndex * sqrtf(std::max(0.f, 1 - cosi * cosi));
  // Total internal reflection
  if (sint >= 1) {
      kr = 1;
  }
  else {
      float cost = sqrtf(std::max(0.f, 1 - sint * sint));
      cosi = fabsf(cosi);
      float Rs = ((refractiveIndex * cosi) - (etai * cost)) / ((refractiveIndex * cosi) + (etai * cost));
      float Rp = ((etai * cosi) - (refractiveIndex * cost)) / ((etai * cosi) + (refractiveIndex * cost));
      kr = (Rs * Rs + Rp * Rp) / 2;
  }
  return kr;
}


bool isInShadow(vec3 pointLightVector, float distanceToLight, RayTriangleIntersection self, float pointLight) {
  for (int i = 0; i < (int)modelTriangles.size(); i++) {
    if (isTriangleSelf(self.intersectedTriangle, modelTriangles[i])) continue;
    vec3 e0 = modelTriangles[i].vertices[1] - modelTriangles[i].vertices[0];
    vec3 e1 = modelTriangles[i].vertices[2] - modelTriangles[i].vertices[0];
    vec3 SLVector = areaLight.pointLights[pointLight].position - modelTriangles[i].vertices[0];
    mat3 DEMatrix(-normalize(pointLightVector), e0, e1);
    vec3 possibleSolution = glm::inverse(DEMatrix) * SLVector;

    if (isInTriangle(possibleSolution) && abs(possibleSolution.x) < abs(distanceToLight) && possibleSolution.x < 0.001f && !isGlass(modelTriangles[i])) {
      return true;
    }
  }
  return false;
}

float softShadow(RayTriangleIntersection self) {
  float shadow = 0;

  for(int i = 0; i < (int)areaLight.pointLights.size(); i++) {
    vec3 pointLightVector = areaLight.pointLights[i].position - self.intersectionPoint;
    float r = length(pointLightVector);
    if(isInShadow(pointLightVector, r, self, i)) shadow += 1.0f;
  }
  return (1 - shadow/areaLight.pointLights.size());
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

void generateObjectMap() {
  for(int i = 0; i < (int)modelTriangles.size(); i++) {
    if (modelTriangles[i].name.compare("portal") == 0) {
      modelTriangles[i].textureVertices[0].y /= 1.2;
      modelTriangles[i].textureVertices[1].y /= 1.2;
      modelTriangles[i].textureVertices[2].y /= 1.2;
    }
    if(objectMap.count(modelTriangles[i].name) == 0) {
      objectMap[modelTriangles[i].name] = vector<int>();
      objectMap[modelTriangles[i].name].push_back(i);
    }
    else {
      objectMap[modelTriangles[i].name].push_back(i);
    }  
  }
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

inline bool isGlass(ModelTriangle triangle) {
  if (triangle.colour.red == 255 && triangle.colour.green == 0 && triangle.colour.blue == 0) return true;
  else return false;
}

inline bool isMirror(ModelTriangle triangle) {
  if (triangle.colour.red == 255 && triangle.colour.green == 0 && triangle.colour.blue == 255) return true;
  else return false;
}

inline bool isPixelOnScreen(int x, int y) {
  if (y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH) return false;
  else return true;
}

inline vec3 getMiddle(ModelTriangle triangle) {
  return (triangle.vertices[0] + triangle.vertices[1] + triangle.vertices[2])/3.0f;
}

bool isFlatBotttomedTriangle(CanvasTriangle triangle) {
  float y0 = round(triangle.vertices[0].y);
  float y1 = round(triangle.vertices[1].y);
  float y2 = round(triangle.vertices[2].y);

  if (y0 == y1 || y0 == y2 || y1 == y2 ) return true;
  else return false;
}

void translateObject(string instruction, string objectName) {
  vector<int> objectIndices = objectMap[objectName];

  if (instruction.compare("LEFT") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0].x -= 0.5;
      modelTriangles[objectIndices[i]].vertices[1].x -= 0.5;
      modelTriangles[objectIndices[i]].vertices[2].x -= 0.5;
    }
  }
  else if (instruction.compare("RIGHT") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0].x += 0.5;
      modelTriangles[objectIndices[i]].vertices[1].x += 0.5;
      modelTriangles[objectIndices[i]].vertices[2].x += 0.5;
    }
  }
  else if (instruction.compare("UP") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0].y += 0.5;
      modelTriangles[objectIndices[i]].vertices[1].y += 0.5;
      modelTriangles[objectIndices[i]].vertices[2].y += 0.5;
    }
  }
  else if (instruction.compare("DOWN") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0].y -= 0.5;
      modelTriangles[objectIndices[i]].vertices[1].y -= 0.5;
      modelTriangles[objectIndices[i]].vertices[2].y -= 0.5;
    }
  }
  else if (instruction.compare("FORWARD") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0].z += 0.5;
      modelTriangles[objectIndices[i]].vertices[1].z += 0.5;
      modelTriangles[objectIndices[i]].vertices[2].z += 0.5;
    }
  }
  else if (instruction.compare("BACK") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0].z -= 0.5;
      modelTriangles[objectIndices[i]].vertices[1].z -= 0.5;
      modelTriangles[objectIndices[i]].vertices[2].z -= 0.5;
    }
  }
}

void rotateObject(string instruction, string objectName) {
  vector<int> objectIndices = objectMap[objectName];
  vec3 pivotPoint = vec3(0.0);

  for (int i = 0; i < (int)objectIndices.size(); i++) {
     pivotPoint += getMiddle(modelTriangles[objectIndices[i]]);
  }
  pivotPoint /= (float)objectIndices.size();

  mat3x3 rotation;

  if (instruction.compare("CLOCKZ") == 0) {
    rotation = mat3x3(cos(-0.5), -sin(-0.5), 0,
                  sin(-0.5), cos(-0.5), 0,
                  0, 0, 1) *
            mat3x3(cos(0), 0, sin(0),
                  0, 1, 0,
                  -sin(0), 0, cos(0)) *
            mat3x3(1, 0, 0,
                  0, cos(0), -sin(0),
                  0, sin(0), cos(0));
  }
  else if (instruction.compare("CLOCKY") == 0) {
    rotation = mat3x3(cos(0), -sin(0), 0,
                  sin(0), cos(0), 0,
                  0, 0, 1) *
            mat3x3(cos(0.5), 0, sin(0.5),
                  0, 1, 0,
                  -sin(0.5), 0, cos(0.5)) *
            mat3x3(1, 0, 0,
                  0, cos(0), -sin(0),
                  0, sin(0), cos(0));
  }

for (int i = 0; i < (int)objectIndices.size(); i++) {
    modelTriangles[objectIndices[i]].vertices[0] = pivotPoint + ((modelTriangles[objectIndices[i]].vertices[0] - pivotPoint) * rotation);
    modelTriangles[objectIndices[i]].vertices[1] = pivotPoint + ((modelTriangles[objectIndices[i]].vertices[1] - pivotPoint) * rotation);
    modelTriangles[objectIndices[i]].vertices[2] = pivotPoint + ((modelTriangles[objectIndices[i]].vertices[2] - pivotPoint) * rotation);
  }
}

void scaleObject(string instruction, string objectName) {
  vector<int> objectIndices = objectMap[objectName];

  if (instruction.compare("GROW") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0] *= 1.5;
      modelTriangles[objectIndices[i]].vertices[1] *= 1.5;
      modelTriangles[objectIndices[i]].vertices[2] *= 1.5;
    }
  }
  else if (instruction.compare("SHRINK") == 0) {
    for (int i = 0; i < (int)objectIndices.size(); i++) {
      modelTriangles[objectIndices[i]].vertices[0] *= 0.5;
      modelTriangles[objectIndices[i]].vertices[1] *= 0.5;
      modelTriangles[objectIndices[i]].vertices[2] *= 0.5;
    }
  }
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
    else if(event.key.keysym.sym == SDLK_4) {
      cout << "SHORT_BOX" << endl;
      bool polling = true;
      while(polling) {
        if(window.pollForInputEvents(&event)) {
          polling = handleObjectEvent(event, "short_box");
        }
      }
    }
  }
}


bool handleObjectEvent(SDL_Event event, string objectName) {
  if(event.key.keysym.sym == SDLK_LEFT) {
    cout << "Moving " << objectName << " LEFT" << endl;
    translateObject("LEFT", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_RIGHT) {
    cout << "Moving " << objectName << " RIGHT" << endl;
    translateObject("RIGHT", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_UP) {
    cout << "Moving " << objectName << " UP" << endl;
    translateObject("UP", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_DOWN) {
    cout << "Moving " << objectName << " DOWN" << endl;
    translateObject("DOWN", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_f) {
    cout << "Moving " << objectName << " FORWARD" << endl;
    translateObject("FORWARD", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_b) {
    cout << "Moving " << objectName << " BACK" << endl;
    translateObject("BACK", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_z) {
    cout << "Rotating " << objectName << " clockwise about Z" << endl;
    rotateObject("CLOCKZ", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_y) {
    cout << "Rotating " << objectName << " clockwise about Y" << endl;
    rotateObject("CLOCKY", objectName);
    update();
    return false;
  }
  else if(event.key.keysym.sym == SDLK_g) {
    cout << "Growing " << objectName << endl;
    scaleObject("GROW", objectName);
    update();
    return false;
  }
  else if (event.key.keysym.sym == SDLK_s) {
    cout << "Shrinking " << objectName << endl;
    scaleObject("SHRINK", objectName);
    update();
    return false;
  }
  else return true;
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
  vector<vec3> vertexNormals;
  vector<vec2> textureVertices;
  vector<vector<uint32_t>> ppmFile;
  string materialFile;
  Colour currentColour;
  string currentName;

  ifstream ifs;
  ifs.open (OBJFile, std::ifstream::in);

  while( ifs.peek() != EOF || ifs.peek() != -1 )
  {
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
    else if (line[0] == 'v' && line[1] == 'n') {
      string* tokens = split(lineString, ' ');
      vertexNormals.push_back(vec3(stof(*(tokens + 1)), stof(*(tokens + 2)), stof(*(tokens + 3))));
    }
    else if (line[0] == 'o') {
      string* tokens = split(lineString, ' ');
      currentName = *(tokens + 1);
      currentName.erase(remove_if(currentName.begin(), currentName.end(), ::isspace), currentName.end());
    }
    else if (line[0] == 'v') {
      string* tokens = split(lineString, ' ');
      vertices.push_back(vec3(stof(*(tokens + 1)), stof(*(tokens + 2)), stof(*(tokens + 3))));
    }
    else if (line[0] == 'f') {
      string* spaceTokens = split(lineString, ' ');
      vector<string> v;
      vector<string> vn;
      vector<string> vt;
      for (int i = 1; i < 4; i++) {
        string* slashTokens = split(*(spaceTokens + i), '/');
        v.push_back(*(slashTokens + 0));
        if (!textureVertices.empty()) vt.push_back(*(slashTokens + 1));
        if (!vertexNormals.empty()) vn.push_back(*(slashTokens + 2));
      }
      if (textureVertices.empty()) {
        if (vertexNormals.empty()) modelTriangles.push_back(ModelTriangle( currentName, vertices[stoi(v[0]) - 1]*scaleFactor,  vertices[stoi(v[1]) - 1]*scaleFactor, vertices[stoi(v[2]) - 1]*scaleFactor, currentColour));
        else modelTriangles.push_back(ModelTriangle( currentName, vertices[stoi(v[0]) - 1]*scaleFactor,  vertices[stoi(v[1]) - 1]*scaleFactor, vertices[stoi(v[2]) - 1]*scaleFactor, vertices[stoi(vn[0]) - 1], vertices[stoi(vn[1]) - 1], vertices[stoi(vn[2]) - 1], currentColour));
      }
      else {
        int ppmWidth = (int)ppmFile[1].size()-1;
        int ppmHeight = (int)ppmFile[0].size()-1;

        vec2 texturePoint0 = vec2(round(textureVertices[stoi(vt[0]) - 1].x * ppmWidth), round(textureVertices[stoi(vt[0]) - 1].y * ppmHeight));
        vec2 texturePoint1 = vec2(round(textureVertices[stoi(vt[1]) - 1].x * ppmWidth), round(textureVertices[stoi(vt[1]) - 1].y * ppmHeight));
        vec2 texturePoint2 = vec2(round(textureVertices[stoi(vt[2]) - 1].x * ppmWidth), round(textureVertices[stoi(vt[2]) - 1].y * ppmHeight));
        modelTriangles.push_back(ModelTriangle( currentName, vertices[stoi(v[0]) - 1]*scaleFactor,  vertices[stoi(v[1]) - 1]*scaleFactor, vertices[stoi(v[2]) - 1]*scaleFactor, texturePoint0, texturePoint1, texturePoint2, textureFiles.size()-1));
      }
    }
  }
}

// Loads a texture from a ppm into a 2D vector
vector<vector<uint32_t>> loadPPM(string file) {
  ifstream ifs;
  ifs.open (file, std::ifstream::in);

  char format[256], widthAndHeight[256], colourChannel[256];//, comment[256];
  widthAndHeight[0] = '#';
  ifs.getline(format, 256);
  while (widthAndHeight[0] == '#') {
    ifs.getline(widthAndHeight, 256);
  }
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
