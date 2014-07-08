// source code from SGI's GLU library (ftp://ftp.freedesktop.org/pub/mesa/glu/)

#ifndef __MM_GLU__
#define __MM_GLU__

#ifdef _WIN32
	#include "windows.h"
#endif

#if ELISE_Darwin
	#include <OpenGL/gl.h>
#else
	#include <GL/gl.h>
#endif

#ifndef GLAPIENTRY
	#if defined(_MSC_VER) || defined(__MINGW32__)
		#define GLAPIENTRY __stdcall
	#else
		#define GLAPIENTRY
	#endif
#endif

GLint GLAPIENTRY
mmUnProject(GLdouble winx, GLdouble winy, GLdouble winz,
            const GLdouble modelMatrix[16], 
            const GLdouble projMatrix[16],
            const GLint viewport[4],
            GLdouble *objx, GLdouble *objy, GLdouble *objz);

GLint GLAPIENTRY
mmProject(GLdouble objx, GLdouble objy, GLdouble objz, 
          const GLdouble modelMatrix[16], 
          const GLdouble projMatrix[16],
          const GLint viewport[4],
          GLdouble *winx, GLdouble *winy, GLdouble *winz);

void GLAPIENTRY
mmLookAt(GLdouble eyex, GLdouble eyey, GLdouble eyez, GLdouble centerx,
         GLdouble centery, GLdouble centerz, GLdouble upx, GLdouble upy,
         GLdouble upz);

#endif
