#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <atlimage.h>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "ParticleSystem.h"
#include "SmokeRenderer.h"
#include "paramgl.h"
#include "GLSLProgram.h"
#include "SmokeShaders.h"
#include "BmpReader.h"
#include "TiffReader.h"

#include "ParticleSystem_cuda.cuh"
#include "math.h"
// CheckRender object for verification
#define MAX_EPSILON_ERROR	10.0f
#define THRESHOLD			0.40f
//#define IMAGE_WIDTH			1150
//#define IMAGE_HEIGHT		1150
//#define GRID_SIZE			512
//#define SCENE_LENTH			4.0f
//每一度对应OpenGL坐标的1
#define SCALE_FACTOR		1.0f
#define HEIGHT_SCALE		0.0001f

uint numParticles = 326*401*16 ;
bool m_AddScene = true;
bool m_renderScene = true;

ParticleSystem *psystem = 0;
SmokeRenderer *renderer = 0;
GLSLProgram *floorProg = 0;

int winWidth = 1024, winHeight = 768;
int g_TotalErrors = 0;

// view params
int ox, oy;
int buttonState = 0;
bool keyDown[256];

//vec2f startLonLat(-18, 0);
int depthTimes = 2;
vec2f startLonLat(120.509, 29.5796);
float profilePointLocation = 31.16055f;
float spatialRes = 0.009;
vec2f DataSize(1800, 2460);
//vec2f DEMStartLonLat(100, 0);
vec2f DEMStartLonLat(119.5333334, 28.82812857);
//float demRes = 0.033333333f;
float demRes = 0.00274658f;

vec3f cameraPos(124.478745, 3.057085,  27.905659);
//vec3f cameraPos(0, -5, -10);
//vec3f cameraRot(24, 338, 0.0);
vec3f cameraRot(-134.399994, -146.399994, 0.0);
vec3f cameraPosLag(cameraPos);
vec3f cameraRotLag(cameraRot);
//这里的cursor代表 粒子生成的中心位置
vec3f cursorPos(0, 1, 0);
vec3f cursorPosLag(cursorPos);

vec3f lightPos(2.0, 10.0, -5.0);

const float inertia = 0.05f;
const float translateSpeed = 0.002f;
const float cursorSpeed = 0.01f;
const float rotateSpeed = 0.2f;
const float walkSpeed = 0.2f;

enum { M_VIEW = 0, M_MOVE_CURSOR, M_MOVE_LIGHT, M_MOVE_PROFILE };
int mode = 0;
int displayMode = (int) SmokeRenderer::VOLUMETRIC;

// QA AutoTest
bool g_bQAReadback = false;

// toggles
bool displayEnabled = true;
bool paused = false;
bool displaySliders = false;
bool wireframe = false;
bool animateEmitter = true;
bool emitterOn = true;
bool sort = true;
//bool displayLightBuffer = false;
bool displayProfile = false;
bool drawVectors = false;
bool doBlur = false;
bool changeSortOnce = false;

float emitterVel = 0.0f;
uint emitterRate = 1000;
float emitterRadius = 0.08;
float emitterSpread = 0.0;
uint emitterIndex = 0;

// simulation parameters
float timestep = 5.0f;
float currentTime = 0.0f;
int currentMonth = 1;

float spriteSize = 0.025f;
float alpha = 0.01f;
float shadowAlpha = 0.02f;
float particleLifetime = (float)numParticles / (float)emitterRate;
vec3f lightColor(1.0f, 1.0f, 1.0f);
vec3f colorAttenuation(0.5f, 1.0f, 1.0f);
float blurRadius = 1.0f;

int numSlices = 128;
int numDisplayedSlices = numSlices;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

float modelView[16];
ParamListGL *params;

BmpReader *m_bmpReader = 0;
GLuint domTex = 0;
GLuint demTex = 0;
GLuint labelTex = 0;

unsigned char *m_demMat;
unsigned int m_VBOVertices = 0;
unsigned int m_VBOTexCoords = 0;
unsigned int m_VertexCount = 0;

typedef struct 														// Vertex Class
{
	float x;													// X Component
	float y;													// Y Component
	float z;													// Z Component
} Vert;

Vert *m_Vertices=0;

typedef struct													// Texture Coordinate Class
{
	float u;													// U Component
	float v;													// V Component
} TexCoord;
TexCoord *m_TexCoords=0;


// initialize particle system
void initParticles(int numParticles, bool bUseVBO, bool bUseGL)
{
	psystem = new ParticleSystem(numParticles, particleLifetime, bUseVBO, bUseGL, startLonLat, spatialRes, DataSize, SCALE_FACTOR, alpha, depthTimes);
	psystem->reset(ParticleSystem::CONFIG_GRID);

	if (bUseVBO)
	{
		renderer = new SmokeRenderer(numParticles);
		renderer->setDisplayMode((SmokeRenderer::DisplayMode) displayMode);
		renderer->setLightTarget(vec3f(0.0, 1.0, 0.0));

		sdkCreateTimer(&timer);
	}
}

void cleanup()
{
	if (psystem)
	{
		delete psystem;
	}

	if (renderer)
	{
		delete renderer;
	}

	if (floorProg)
	{
		delete floorProg;
	}

	sdkDeleteTimer(&timer);

	if (params)
	{
		delete params;
	}

	if (domTex)
	{
		glDeleteTextures(1, &domTex);
	}
}

//学习一下shader的disable方法
void renderScene()
{
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	if (m_AddScene && m_renderScene && true)
	{
		glColor4f(1.0, 1.0, 1.0, 1.0);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// draw floor
		floorProg->enable();
		floorProg->bindTexture("dom", domTex, GL_TEXTURE_2D, 0);
		//floorProg->bindTexture("label", labelTex, GL_TEXTURE_2D, 1);

		glNormal3f(0.0, 1.0, 0.0);

		// 使用顶点，纹理坐标数组
		glEnableClientState( GL_VERTEX_ARRAY ); 
		glEnableClientState( GL_TEXTURE_COORD_ARRAY ); 

		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_VBOVertices );
		glVertexPointer( 3, GL_FLOAT, 0, (char *) NULL );		// 设置顶点数组的指针为顶点缓存
		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_VBOTexCoords );
		glTexCoordPointer( 2, GL_FLOAT, 0, (char *) NULL );		// 设置顶点数组的指针为纹理坐标缓存

		// 渲染
		glDrawArrays( GL_TRIANGLES, 0, m_VertexCount );		

		glDisableClientState( GL_VERTEX_ARRAY );	
		glDisableClientState( GL_TEXTURE_COORD_ARRAY );

		floorProg->disable();
		glDisable(GL_BLEND);
	}

	// set shadow matrix as texture matrix 
	// ??
	matrix4f shadowMatrix = renderer->getShadowMatrix();
	glActiveTexture(GL_TEXTURE0);
	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();

	//draw profile剖面
	

	//// draw light
	//glMatrixMode(GL_MODELVIEW);
	//glPushMatrix();
	//glTranslatef(lightPos.x, lightPos.y, lightPos.z);
	//glColor3fv(&lightColor[0]);
	//glutSolidSphere(0.1, 10, 5);
	//glPopMatrix();
}

void renderLabel()
{
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
	glEnable(GL_BLEND);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(0.0, (GLfloat)(glutGet(GLUT_WINDOW_HEIGHT) - 1.0), 0.0);
	glScalef(1.0, -1.0, 1.0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT), -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glColor3f(0.8f, 0.1f, 0.1f);
	glPrint(winWidth / 2 - 190, 20, "TEMPERATURE DATA SPATIO-TEMPORAL PROCESS", (void *) GLUT_BITMAP_HELVETICA_18);

	std::stringstream timeString;
	timeString << "From 2000-" << currentMonth / 6 + 1<< "-" << (currentMonth % 6 * 5 + 2.5) << " to " << "2000-" << currentMonth / 6 + 1<< "-" << (currentMonth % 6 * 5 + 7.5);
	glPrint(winWidth / 2 - 140, 50, timeString.str().data(), (void *) GLUT_BITMAP_HELVETICA_18);

	/*std::stringstream cameraPosString;
	cameraPosString << "camera-x:" << cameraPos.x << " camera-y:" << cameraPos.y << " camera-z:" <<cameraPos.z;
	glPrint(winWidth / 2 - 210, 80, cameraPosString.str().data(), (void *) GLUT_BITMAP_HELVETICA_18);*/

	/*std::stringstream cameraRotateString;
	cameraRotateString << "cameraRot-x:" << cameraRot.x << " cameraRot-y:" << cameraRot.y << " cameraRot-z:" <<cameraRot.z;
	glPrint(winWidth / 2 - 220, 110, cameraRotateString.str().data(), (void *) GLUT_BITMAP_HELVETICA_18);*/

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
}

// main rendering loop
//看完论文后看如何基于纹理绘制Volume的
void display()
{
	sdkStartTimer(&timer);

	// move camera
	//if (cameraPos[1] > 0.0f)
	//{
	//	cameraPos[1] = 0.0f;
	//}

	//惯性:实现方法，变化后的-变化前的 *参数，慢慢变。。
	cameraPosLag += (cameraPos - cameraPosLag) * inertia;
	cameraRotLag += (cameraRot - cameraRotLag) * inertia;
	cursorPosLag += (cursorPos - cursorPosLag) * inertia;
	
	// view transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRotatef(cameraRotLag[0], 1.0, 0.0, 0.0);
	glRotatef(cameraRotLag[1], 0.0, 1.0, 0.0);
	glTranslatef(cameraPosLag[0], cameraPosLag[1], cameraPosLag[2]);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	// update the simulation
	if (!paused)
	{
		if (emitterOn)
		{
			//runEmitter();
		}

		SimParams &p = psystem->getParams();

		psystem->step(timestep, &currentTime, &currentMonth);
		currentTime += timestep;
	}

	//前面部分是生成粒子及粒子的代谢
	//后面部分对当前粒子场景进行基于纹理的体绘制

	//计算halfvector，作为粒子排序轴
	renderer->calcVectors();
	vec3f sortVector = renderer->getSortVector();

	psystem->setSortVector(make_float3(sortVector.x, sortVector.y, sortVector.z));
	psystem->setModelView(modelView);
	psystem->setSorting(sort);
	psystem->depthSort();

	// render
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if (m_AddScene)
	{
		renderScene();
		//m_renderScene = false;
	}

	// draw particles
	if (displayEnabled)
	{
		//裁剪到场景上方
		//m_imageFbo完成粒子合成结果
		renderer->beginSceneRender(SmokeRenderer::SCENE_BUFFER);
		renderScene();
		renderer->endSceneRender(SmokeRenderer::SCENE_BUFFER);

		// lightbuffer叠加光照效果和倒影
		renderer->beginSceneRender(SmokeRenderer::LIGHT_BUFFER);
		renderScene();
		renderer->endSceneRender(SmokeRenderer::LIGHT_BUFFER);

		m_renderScene = true;

		renderer->setPositionBuffer(psystem->getPosBuffer());
		//renderer->setVelocityBuffer(psystem->getVelBuffer());
		renderer->setColorBuffer(psystem->getColorBuffer());
		renderer->setIndexBuffer(psystem->getSortedIndexBuffer());
		renderer->setNumParticles(psystem->getNumParticles());
		renderer->setParticleRadius(spriteSize);
		//renderer->setDisplayLightBuffer(displayLightBuffer);
		renderer->setDisplayProfile(displayProfile);
		renderer->setProfileLocation(profilePointLocation);
		renderer->setAlpha(alpha);
		renderer->setShadowAlpha(shadowAlpha);
		renderer->setLightPosition(lightPos);
		renderer->setColorAttenuation(colorAttenuation);
		renderer->setLightColor(lightColor);
		renderer->setNumSlices(numSlices);
		renderer->setNumDisplayedSlices(numDisplayedSlices);
		renderer->setBlurRadius(blurRadius);

		renderer->render();

		if (drawVectors)
		{
			renderer->debugVectors();
		}

		//renderer->drawProfile();
	}

	//初始化完了就不sort，提高速度
	if (!changeSortOnce)
	{
		sort = false;
		changeSortOnce = true;
	}

	// display sliders
	if (displaySliders)
	{
		glDisable(GL_DEPTH_TEST);
		glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
		glEnable(GL_BLEND);
		params->Render(0, 0);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}
	renderLabel();

	glutSwapBuffers();
	glutReportErrors();
	sdkStopTimer(&timer);

	fpsCount++;

	// this displays the frame rate updated every second (independent of frame rate)
	if (fpsCount >= fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "MARINE ENVIRONMENTAL DATA SPATIO-TEMPORAL SIMULATION (%d particles): %3.1f fps.", numParticles, ifps);
		//sprintf(fps, "pos:%3.6f %3.6f %3.6f rot:%3.6f %3.6f %3.6f", cameraPos[0],cameraPos[1],cameraPos[2], cameraRot[0],cameraRot[1],cameraRot[2]);
		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

		if (paused)
		{
			fpsLimit = 0;
		}

		sdkResetTimer(&timer);
	}
}

// GLUT callback functions
void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float) w / (float) h, 0.01, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	renderer->setFOV(60.0);
	renderer->setWindowSize(w, h);
}

//鼠标按键
void mouse(int button, int state, int x, int y)
{
	int mods;

	if (state == GLUT_DOWN)
	{
		buttonState |= 1<<button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	mods = glutGetModifiers();

	if (mods & GLUT_ACTIVE_SHIFT)
	{
		buttonState = 2;
	}
	else if (mods & GLUT_ACTIVE_CTRL)
	{
		buttonState = 3;
	}

	ox = x;
	oy = y;

	if (displaySliders)
	{
		if (params->Mouse(x, y, button, state))
		{
			glutPostRedisplay();
			return;
		}
	}

	glutPostRedisplay();
}

// transfrom vector by matrix
void xform(vec3f &v, vec3f &r, float *m)
{
	r.x = v.x*m[0] + v.y*m[4] + v.z*m[8] + m[12];
	r.y = v.x*m[1] + v.y*m[5] + v.z*m[9] + m[13];
	r.z = v.x*m[2] + v.y*m[6] + v.z*m[10] + m[14];
}

// transform vector by transpose of matrix (assuming orthonormal)
// 平移变换的增量，v为方向，m为ModelviewMatrix
void ixform(vec3f &v, vec3f &r, float *m)
{
	r.x = v.x*m[0] + v.y*m[1] + v.z*m[2];
	r.y = v.x*m[4] + v.y*m[5] + v.z*m[6];
	r.z = v.x*m[8] + v.y*m[9] + v.z*m[10];
}

void ixformPoint(vec3f &v, vec3f &r, float *m)
{
	vec3f x;
	x.x = v.x - m[12];
	x.y = v.y - m[13];
	x.z = v.z - m[14];
	ixform(x, r, m);
}

//鼠标移动
void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (displaySliders)
	{
		if (params->Motion(x, y))
		{
			ox = x;
			oy = y;
			glutPostRedisplay();
			return;
		}
	}

	switch (mode)
	{
	case M_VIEW:
		{
			if (buttonState == 1)
			{
				// left = rotate
				cameraRot[0] += dy * rotateSpeed;
				cameraRot[1] -= dx * rotateSpeed;
			}

			if (buttonState == 2)
			{
				// middle = translate
				vec3f v = vec3f(dx*translateSpeed, -dy*translateSpeed, 0.0f);
				vec3f r;
				ixform(v, r, modelView);
				cameraPos += r;
			}

			if (buttonState == 3)
			{
				// left+ctrl = zoom
				vec3f v = vec3f(0.0, 0.0, dy*translateSpeed);
				vec3f r;
				ixform(v, r, modelView);
				cameraPos += r;
			}
		}
		break;

	case M_MOVE_CURSOR:
		{
			if (buttonState==1)
			{
				vec3f v = vec3f(dx*cursorSpeed, -dy*cursorSpeed, 0.0f);
				vec3f r;
				ixform(v, r, modelView);
				cursorPos += r;
			}
			else if (buttonState==2)
			{
				vec3f v = vec3f(0.0f, 0.0f, dy*cursorSpeed);
				vec3f r;
				ixform(v, r, modelView);
				cursorPos += r;
			}
		}
		break;

	case M_MOVE_LIGHT:
		if (buttonState==1)
		{
			vec3f v = vec3f(dx*cursorSpeed, -dy*cursorSpeed, 0.0f);
			vec3f r;
			ixform(v, r, modelView);
			lightPos += r;
		}
		else if (buttonState==2)
		{
			vec3f v = vec3f(0.0f, 0.0f, dy*cursorSpeed);
			vec3f r;
			ixform(v, r, modelView);
			lightPos += r;
		}

		break;

	case M_MOVE_PROFILE:
		if (buttonState==1)
		{
			//vec3f v = vec3f(dx*cursorSpeed, -dy*cursorSpeed, 0.0f);
			//vec3f r;
			//ixform(v, r, modelView);
			profilePointLocation += dx*cursorSpeed;
			if(profilePointLocation>32.7415)
				profilePointLocation = 32.7415;
			if(profilePointLocation<29.5796)
				profilePointLocation = 29.5796;
		}

	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case ' ':
		paused = !paused;
		break;

	case 13:
		psystem->step(timestep, &currentTime, &currentMonth);
		renderer->setPositionBuffer(psystem->getPosBuffer());
		//renderer->setVelocityBuffer(psystem->getVelBuffer());
		renderer->setColorBuffer(psystem->getColorBuffer());
		break;

	case '\033':
		cleanup();
		exit(EXIT_SUCCESS);
		break;

	case 'v':
		mode = M_VIEW;
		animateEmitter = true;
		break;

	case 'm':
		mode = M_MOVE_CURSOR;
		animateEmitter = false;
		break;

	case 'l':
		mode = M_MOVE_LIGHT;
		break;

	case 'r':
		displayEnabled = !displayEnabled;
		break;

	case '1':
		psystem->reset(ParticleSystem::CONFIG_GRID);
		break;

	case '2':
		emitterOn ^= 1;
		break;

	case 'W':
		wireframe = !wireframe;
		break;

	case 'h':
		displaySliders = !displaySliders;
		break;

	case 'o':
		sort ^= 1;
		psystem->setSorting(sort);
		break;

	case 'D':
		displayProfile ^= 1;
		mode = M_MOVE_PROFILE;
		break;

	case 'p':
		displayMode = (displayMode + 1) % SmokeRenderer::NUM_MODES;
		renderer->setDisplayMode((SmokeRenderer::DisplayMode) displayMode);
		break;

	case 'P':
		displayMode--;

		if (displayMode < 0)
		{
			displayMode = SmokeRenderer::NUM_MODES - 1;
		}

		renderer->setDisplayMode((SmokeRenderer::DisplayMode) displayMode);
		break;

	case 'V':
		drawVectors ^= 1;
		break;

	case '=':
		numSlices *= 2;

		if (numSlices > 256)
		{
			numSlices = 256;
		}

		numDisplayedSlices = numSlices;
		break;

	case '-':
		if (numSlices > 1)
		{
			numSlices /= 2;
		}

		numDisplayedSlices = numSlices;
		break;

	case 'b':
		doBlur ^= 1;
		renderer->setDoBlur(doBlur);
		break;
	}

	printf("numSlices = %d\n", numSlices);
	keyDown[key] = true;

	glutPostRedisplay();
}
//up了，就停止按下时的操作====》改建置为false
void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
	keyDown[key] = false;
}

void special(int k, int x, int y)
{
	if (displaySliders)
	{
		params->Special(k, x, y);
	}
}

void idle(void)
{
	// move camera in view direction
	/*
	0   4   8   12  x
	1   5   9   13  y
	2   6   10  14  z
	*/
	if (keyDown['w'])
	{
		cameraPos[0] += modelView[2] * walkSpeed;
		cameraPos[1] += modelView[6] * walkSpeed;
		cameraPos[2] += modelView[10] * walkSpeed;
	}

	if (keyDown['s'])
	{
		cameraPos[0] -= modelView[2] * walkSpeed;
		cameraPos[1] -= modelView[6] * walkSpeed;
		cameraPos[2] -= modelView[10] * walkSpeed;
	}

	if (keyDown['a'])
	{
		cameraPos[0] += modelView[0] * walkSpeed;
		cameraPos[1] += modelView[4] * walkSpeed;
		cameraPos[2] += modelView[8] * walkSpeed;
	}

	if (keyDown['d'])
	{
		cameraPos[0] -= modelView[0] * walkSpeed;
		cameraPos[1] -= modelView[4] * walkSpeed;
		cameraPos[2] -= modelView[8] * walkSpeed;
	}

	if (keyDown['e'])
	{
		cameraPos[0] += modelView[1] * walkSpeed;
		cameraPos[1] += modelView[5] * walkSpeed;
		cameraPos[2] += modelView[9] * walkSpeed;
	}

	if (keyDown['q'])
	{
		cameraPos[0] -= modelView[1] * walkSpeed;
		cameraPos[1] -= modelView[5] * walkSpeed;
		cameraPos[2] -= modelView[9] * walkSpeed;
	}

	//此处不再模拟轨迹运动
	//if (animateEmitter)
	//{
	//	const float speed = 0.02f;
	//	cursorPos.x = sin(currentTime*speed)*1.5f;
	//	cursorPos.y = 0.5f + sin(currentTime*speed*1.3f);
	//	cursorPos.z = cos(currentTime*speed)*1.5f;
	//}

	glutPostRedisplay();
}

// initialize sliders
void initParams()
{
	// create a new parameter list
	params = new ParamListGL("misc");

	params->AddParam(new Param<int>("displayed slices", numDisplayedSlices, 0, 256, 1, &numDisplayedSlices));

	params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.001f, &timestep));

	SimParams &p = psystem->getParams();
	params->AddParam(new Param<float>("damping", 0.99f, 0.0f, 1.0f, 0.001f, &p.globalDamping));
	params->AddParam(new Param<float>("gravity", 0.00000f, 0.001f, -0.001f, 0.00001f, &p.gravity.y));

	params->AddParam(new Param<float>("noise freq", 0.1f, 0.0f, 1.0f, 0.001f, &p.noiseFreq));
	params->AddParam(new Param<float>("noise strength", 0.00002f, 0.0f, 0.0001f, 0.00001f, &p.noiseAmp));
	params->AddParam(new Param<float>("noise anim", 0.0f, -0.001f, 0.001f, 0.0001f, &p.noiseSpeed.y));

	params->AddParam(new Param<float>("sprite size", spriteSize, 0.0f, 0.1f, 0.001f, &spriteSize));
	params->AddParam(new Param<float>("alpha", alpha, 0.0f, 1.0f, 0.001f, &alpha));

	params->AddParam(new Param<float>("light color r", lightColor[0], 0.0f, 1.0f, 0.01f, &lightColor[0]));
	params->AddParam(new Param<float>("light color g", lightColor[1], 0.0f, 1.0f, 0.01f, &lightColor[1]));
	params->AddParam(new Param<float>("light color b", lightColor[2], 0.0f, 1.0f, 0.01f, &lightColor[2]));

	params->AddParam(new Param<float>("atten color r", colorAttenuation[0], 0.0f, 1.0f, 0.01f, &colorAttenuation[0]));
	params->AddParam(new Param<float>("atten color g", colorAttenuation[1], 0.0f, 1.0f, 0.01f, &colorAttenuation[1]));
	params->AddParam(new Param<float>("atten color b", colorAttenuation[2], 0.0f, 1.0f, 0.01f, &colorAttenuation[2]));
	params->AddParam(new Param<float>("shadow alpha", shadowAlpha, 0.0f, 0.1f, 0.001f, &shadowAlpha));

	params->AddParam(new Param<float>("blur radius", blurRadius, 0.0f, 10.0f, 0.1f, &blurRadius));

	params->AddParam(new Param<float>("emitter radius", emitterRadius, 0.0f, 2.0f, 0.01f, &emitterRadius));
	params->AddParam(new Param<uint>("emitter rate", emitterRate, 0, 10000, 1, &emitterRate));
	params->AddParam(new Param<float>("emitter velocity", emitterVel, 0.0f, 0.1f, 0.001f, &emitterVel));
	params->AddParam(new Param<float>("emitter spread", emitterSpread, 0.0f, 0.1f, 0.001f, &emitterSpread));

	params->AddParam(new Param<float>("particle lifetime", particleLifetime, 0.0f, 1000.0f, 1.0f, &particleLifetime));
}

//调用按键响应
void mainMenu(int i)
{
	key((unsigned char) i, 0, 0);
}
//右键菜单
void initMenus()
{
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Reset block [1]", '1');
	glutAddMenuEntry("Toggle emitter [2]", '2');
	glutAddMenuEntry("Toggle animation [ ]", ' ');
	glutAddMenuEntry("Step animation [ret]", 13);
	glutAddMenuEntry("View mode [v]", 'v');
	glutAddMenuEntry("Move cursor mode [m]", 'm');
	glutAddMenuEntry("Move light mode [l]", 'l');
	glutAddMenuEntry("Toggle point rendering [p]", 'p');
	glutAddMenuEntry("Toggle sliders [h]", 'h');
	glutAddMenuEntry("Toggle sorting [o]", 'o');
	glutAddMenuEntry("Toggle vectors [V]", 'V');
	glutAddMenuEntry("Display profile buffer [D]", 'D');
	glutAddMenuEntry("Toggle shadow blur [b]", 'b');
	glutAddMenuEntry("Increase no. slices [=]", '=');
	glutAddMenuEntry("Decrease no. slices [-]", '-');
	glutAddMenuEntry("Quit (esc)", '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
{
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(target, tex);
	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
	return tex;
}

GLuint load24bitTexture(char *filename)
{
	unsigned int *data = 0;
	unsigned int width, height;
	//sdkLoadPPM4ub(filename, &data, &width, &height);
	if (!m_bmpReader)
	{
		m_bmpReader = new BmpReader();
	}
	
	//读24bit的bmp
	m_bmpReader->Read24bitBmp((uchar4 **)&data, &width, &height, filename);

	if (!data)
	{
		printf("Error opening file '%s'\n", filename);
		return 0;
	}

	printf("Loaded '%s', %d x %d pixels\n", filename, width, height);

	return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
}

GLuint ATLLoadTexture(const char *fileName)  
{  
	BITMAP bm;  
	GLuint idTexture = 0;  
	CImage img;             //需要头文件atlimage.h  
	HRESULT hr = img.Load(fileName);  
	if ( !SUCCEEDED(hr) )   //文件加载失败  
	{  
		MessageBox(NULL, "文件加载失败", "ERROR", 0);  
		return NULL;  
	}  
	HBITMAP hbmp = img;  
	if(!GetObject(hbmp, sizeof(bm), &bm))  
	{
		return 0;  
	}

	return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_BGR, bm.bmWidth, bm.bmHeight, bm.bmBits);	
} 

//用GDAL读取DEM
//将DEM输入到顶点数组
bool InitDEM()
{	
	char *imagePath = "data/image/Dem2.tif";

	//读TIFF图像及长宽
	TiffReader *tfReader = new TiffReader();
	tfReader->OpenFile(imagePath);
	unsigned int height = tfReader->GetRasterHeight();
	unsigned int width = tfReader->GetRasterWidth();

	//总顶点数
	m_VertexCount = width * height * 6;
	m_Vertices = new Vert[m_VertexCount];
	m_TexCoords = new TexCoord[m_VertexCount];
	unsigned int flResolution = 1;
	float scaleUnit = SCALE_FACTOR;

	unsigned int nX, nZ, nTri, nIndex=0;									
	unsigned int flX, flZ=0;
	for( nZ = 0; nZ < height - flResolution; nZ++ )
	{
		for( nX = 0; nX < width - flResolution; nX++ )
		{
			for( nTri = 0; nTri < 6; nTri++ )
			{
				flX = nX + ( ( nTri == 1 || nTri == 2 || nTri == 5 ) ? flResolution : 0 );
				flZ = nZ + ( ( nTri == 2 || nTri == 4 || nTri == 5 ) ? flResolution : 0 );

				m_Vertices[nIndex].x = -DEMStartLonLat.x  -flX * scaleUnit * demRes;
				m_Vertices[nIndex].y = tfReader->GetPixelValue(flX, flZ) *  HEIGHT_SCALE;
				m_Vertices[nIndex].y = m_Vertices[nIndex].y > 0 ? m_Vertices[nIndex].y : 0;
				m_Vertices[nIndex].y = -m_Vertices[nIndex].y;
				//我也不知道为什么Y要翻过来，但是这样刚刚正好
				//好像整个视角都是翻过来的，先凑合着用吧
				//m_Vertices[nIndex].z = -DEMStartLonLat.y - (height - flZ) * scaleUnit * demRes;
				m_Vertices[nIndex].z = -DEMStartLonLat.y -  (height-flZ) * scaleUnit * demRes;

				m_TexCoords[nIndex].u = flX *1.0f / width;
				m_TexCoords[nIndex].v = flZ *1.0f / height;

				nIndex++;
			}
		}
	}

	glGenBuffersARB( 1, &m_VBOVertices );	// 创建一个顶点缓存，并把顶点数据绑定到缓存	
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_VBOVertices );				
	glBufferDataARB( GL_ARRAY_BUFFER_ARB, m_VertexCount*3*sizeof(float), m_Vertices, GL_STATIC_DRAW_ARB );
	glGenBuffersARB( 1, &m_VBOTexCoords ); // 创建一个纹理缓存，并把纹理数据绑定到缓存
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_VBOTexCoords ); 
	glBufferDataARB( GL_ARRAY_BUFFER_ARB, m_VertexCount*2*sizeof(float), m_TexCoords, GL_STATIC_DRAW_ARB );

	// 删除分配的内存
	delete [] m_Vertices; m_Vertices = NULL;
	delete [] m_TexCoords; m_TexCoords = NULL;
	delete tfReader;

	return true;
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(50, 50);
	glutInitWindowSize(winWidth, winHeight);
	glutCreateWindow("CO2 FLUX SIMULATION");
	glewInit();

	if (wglewIsSupported("WGL_EXT_swap_control"))
	{
		// disable vertical sync
		wglSwapIntervalEXT(0);
	}

	glEnable(GL_DEPTH_TEST);

	// load floor texture
	if (m_AddScene && true)
	{
		char *imagePath = sdkFindFilePath("data/image/Dom2.bmp", argv[0]);
		domTex = load24bitTexture(imagePath);
		


		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);


		//imagePath = sdkFindFilePath("LABEL.bmp", "");
		//labelTex = load24bitTexture(imagePath);

		//创建顶点数组
		InitDEM();

		floorProg = new GLSLProgram(floorVS, floorPS);
	}

	glutReportErrors();
}


int main(int argc, char **argv)
{
	initGL(&argc, argv);
	findCudaGLDevice(argc, (const char **)argv);

	// This is the normal code path for SmokeParticles
	initParticles(numParticles, true, true);
	initParams();
	initMenus();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutKeyboardUpFunc(keyUp);
	glutSpecialFunc(special);
	glutIdleFunc(idle);

	glutMainLoop();

	cudaDeviceReset();
	exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);


	
}
