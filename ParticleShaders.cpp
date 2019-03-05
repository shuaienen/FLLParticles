#define STRINGIFY(A) #A

#pragma region 点模式
// particle vertex shader
const char *particleVS = STRINGIFY(
                             uniform float pointRadius;  // point size in world space    \n
                             uniform float pointScale;   // scale to calculate size in pixels \n
                             uniform vec4 eyePos;                                        \n
                             void main()                                                 \n
{
    vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                   \n
    gl_Position = gl_ModelViewProjectionMatrix *wpos;      \n

    vec4 eyeSpacePos = gl_ModelViewMatrix *wpos;           \n
    float dist = length(eyeSpacePos.xyz);                   \n
    gl_PointSize = pointRadius * (pointScale / dist);       \n


	gl_FrontColor = gl_Color;                               \n
}                                                           \n
                         );

const char *simplePS = STRINGIFY(
		void main()                                            \n
{
		gl_FragColor = gl_Color;                               \n
}                                                              \n
	);
#pragma endregion 


#pragma region 图片模式的PS
// render particle without shadows
const char *particlePS = STRINGIFY(
                             void main()							\n
{
    \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n

    N.z = sqrt(1.0-r2);                                            \n

    float alpha = clamp((1.0 - r2), 0.0, 1.0);                     \n									\n
	
	alpha = gl_Color.w == 0.0?0.0:0.5;

    gl_FragColor = vec4(gl_Color.x * alpha, gl_Color.y * alpha, gl_Color.z * alpha, alpha);              \n
}                                                                  \n
                         );
#pragma endregion 

	//EmitVertex()这个function代表将之前所有Varying out变量(包含内建的与自己声明的)依照现在的值作为一个Vertex的数据输出.

	//EmitPrimitive()将之前所有调用EmitVertex()的Vertex组成一个Primitive.

#pragma region slice模式
	// motion blur shaders
	const char *mblurVS = STRINGIFY(
		uniform float timestep;										\n
		void main()													\n
	{
		\n
			// SmokeRenderer::drawPoints里设定的
			vec3 pos    = gl_Vertex.xyz;                           \n


			gl_Position    = gl_ModelViewMatrix * vec4(pos, 1.0);  \n // eye space

			gl_TexCoord[1].x = 1.0;                                \n
			gl_FrontColor = vec4(gl_Color.xyz, gl_Color.w);     \n
	}                                                            \n
	);

		// motion blur geometry shader
		// - outputs stretched quad between previous and current positions
		// 点转图元
		const char *mblurGS =
			"#version 120\n"
			"#extension GL_EXT_geometry_shader4 : enable\n"
			STRINGIFY(
			uniform float pointRadius;  // point size in world space       \n
		void main()                                                    \n
		{
			\n
				// aging                                                   \n
				float phase = gl_TexCoordIn[0][1].x;                       \n
				float radius = pointRadius;                                \n

				// eye space                                               \n
				vec3 pos = gl_PositionIn[0].xyz;                           \n
				vec3 pos2 = gl_TexCoordIn[0][0].xyz;                       \n
				vec3 motion = pos - pos2;                                  \n
				//vec3 motion = vec3(1,1,1);
				vec3 dir = normalize(motion);                              \n
				float len = length(motion);                                \n

				vec3 x = dir *radius;                                     \n
				vec3 view = normalize(-pos);                               \n
				vec3 y = normalize(cross(dir, view)) * radius;             \n
				float facing = dot(view, dir);                             \n

				// check for very small motion to avoid jitter             \n
				float threshold = 0.01;                                    \n

				if ((len < threshold) || (facing > 0.95) || (facing < -0.95))
				{
					\n
						pos2 = pos;
					\n
						x = vec3(radius, 0.0, 0.0);
					\n
						y = vec3(0.0, -radius, 0.0);
					\n
				}                                                          \n

				// output quad                                             \n
				gl_FrontColor = gl_FrontColorIn[0];                        \n
				gl_TexCoord[0] = vec4(0, 0, 0, phase);                     \n
				gl_TexCoord[1] = gl_PositionIn[0];                         \n
				gl_Position = gl_ProjectionMatrix * vec4(pos + x + y, 1);  \n
				EmitVertex();                                              \n
				gl_FrontColor = gl_FrontColorIn[0];    
				gl_TexCoord[0] = vec4(0, 1, 0, phase);                     \n
				gl_TexCoord[1] = gl_PositionIn[0];                         \n
				gl_Position = gl_ProjectionMatrix * vec4(pos + x - y, 1);  \n
				EmitVertex();                                              \n
				gl_FrontColor = gl_FrontColorIn[0];    
				gl_TexCoord[0] = vec4(1, 0, 0, phase);                     \n
				gl_TexCoord[1] = gl_PositionIn[0];                         \n
				gl_Position = gl_ProjectionMatrix * vec4(pos2 - x + y, 1); \n
				EmitVertex();                                              \n
				gl_FrontColor = gl_FrontColorIn[0];    
				gl_TexCoord[0] = vec4(1, 1, 0, phase);                     \n
				gl_TexCoord[1] = gl_PositionIn[0];                         \n
				gl_Position = gl_ProjectionMatrix * vec4(pos2 - x - y, 1); \n
				EmitVertex();                                              \n
		}                                                              \n
		);

// render particle including shadows
const char *particleShadowPS = STRINGIFY(
                                   uniform float pointRadius;                                         \n
                                   uniform sampler2D shadowTex;                                       \n
								   //uniform sampler2D tex;                                             \n
                                   void main()                                                        \n
{
    \n
    // calculate eye-space sphere normal from texture coordinates  \n
    vec3 N;                                                        \n
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
    float r2 = dot(N.xy, N.xy);                                    \n
	if(gl_Color.w == 0.0) discard;
    if (r2 > 1.0) discard;                                         \n // kill pixels outside circle
    N.z = sqrt(1.0-r2);                                            \n
	//以上。。把原本方的纹理搞成了圆的

    vec4 eyeSpacePos = gl_TexCoord[1];                             \n
    vec4 eyeSpaceSpherePos = vec4(eyeSpacePos.xyz + N*pointRadius, 1.0); \n // point on sphere
    vec4 shadowPos = gl_TextureMatrix[0] * eyeSpaceSpherePos;      \n
    vec3 shadow = vec3(1.0) - texture2DProj(shadowTex, shadowPos.xyw).xyz;  \n
	//shadow = vec3(1.0); \n

    //  float alpha = saturate(1.0 - r2);                              \n
	//边缘透明
    float alpha = clamp((1.0 - r2), 0.0, 1.0);                     \n
    alpha *= gl_Color.w;                                           \n

    gl_FragColor = vec4(gl_Color.xyz *shadow * alpha, alpha);     \n  // premul alpha
}
                               );
#pragma endregion



#pragma region SLICE模式中是否抖动光线
const char *passThruVS = STRINGIFY(
		void main()                                                        \n
{
	\n
		gl_Position = gl_Vertex;                                       \n
		gl_TexCoord[0] = gl_MultiTexCoord0;                            \n
		gl_FrontColor = gl_Color;                                      \n
}                                                                  \n
);


// 4 tap 3x3 gaussian blur
const char *blurPS = STRINGIFY(
                         uniform sampler2D tex;                                                                \n
                         uniform vec2 texelSize;                                                               \n
                         uniform float blurRadius;                                                             \n
                         void main()                                                                           \n
{
    \n
    vec4 c;                                                                           \n
    c = texture2D(tex, gl_TexCoord[0].xy + vec2(-0.5, -0.5)*texelSize*blurRadius);    \n
    c += texture2D(tex, gl_TexCoord[0].xy + vec2(0.5, -0.5)*texelSize*blurRadius);    \n
    c += texture2D(tex, gl_TexCoord[0].xy + vec2(0.5, 0.5)*texelSize*blurRadius);     \n
    c += texture2D(tex, gl_TexCoord[0].xy + vec2(-0.5, 0.5)*texelSize*blurRadius);    \n
    c *= 0.25;                                                                        \n

    gl_FragColor = c;                                                                 \n
}                                                                                     \n
                     );
#pragma endregion 


#pragma region 渲染背景
// floor shader
const char *floorVS = STRINGIFY(
                          varying vec4 vertexPosEye;								\n // vertex position in eye space
                          varying vec3 normalEye;                                      \n
                          void main()                                                  \n
{
    \n
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;  \n
    gl_TexCoord[0] = gl_MultiTexCoord0;                      \n
    vertexPosEye = gl_ModelViewMatrix *gl_Vertex;           \n
    normalEye = gl_NormalMatrix *gl_Normal;                 \n
    gl_FrontColor = gl_Color;                                \n
}                                                            \n
                      );



const char *floorPS = STRINGIFY(
		uniform sampler2D dom;															\n
		uniform sampler2D label;														\n
		varying vec4 vertexPosEye;								                   \n // vertex position in eye space
		varying vec3 normalEye;                                                       \n
		void main()                                                                   \n
	{
		\n
			if(gl_TexCoord[0].x>0.995||gl_TexCoord[0].x<0.005||gl_TexCoord[0].y>0.995||gl_TexCoord[0].y<0.005)
				discard;;
			vec4 colorMap = texture2D(dom, gl_TexCoord[0].xy);									\n
			colorMap.w = 1.0;																\n
			if (colorMap.x > 0.95 && colorMap.y > 0.95 &&colorMap.z > 0.95)		
				discard;
			//非白色处，设置为标注
			vec4 colorLabel = texture2D(label, gl_TexCoord[0].xy);								\n
			if (colorLabel.x < 0.95 && colorLabel.y < 0.95 &&colorLabel.z < 0.95)				\n
			{																					 \n
				colorMap = vec4(colorLabel.x, colorLabel.y, colorLabel.z, 1.0);					\n
			}																					\n

			gl_FragColor = vec4(colorMap.xyz, gl_Color * colorMap.w);					\n
	}																					\n
	);
#pragma endregion 
