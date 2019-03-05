#include "GlExtensions.h"
#include "general/CMake_defines.h"

#include <list>
#include <algorithm>
#include <iomanip>

#if ELISE_windows
	#include <windows.h> 
#endif

#if ELISE_Darwin
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
#endif

using namespace std;

void parseExtensions(const string &i_extensionsString, vector<string> &o_extensionsVector)
{
	// parse the string and put tokens into a sorted list
	list<string> extensionsList;
	size_t pos0 = 0, pos1;
	bool ok = true;
	while (ok)
	{
		pos1 = i_extensionsString.find(' ', pos0);
		if (pos1 == string::npos)
		{
			pos1 = i_extensionsString.length();
			ok = false;
		}
		if (pos0 != pos1) extensionsList.push_back(i_extensionsString.substr(pos0, pos1 - pos0 ));
		pos0 = pos1 + 1;
	}
	extensionsList.sort();

	// copy the list into a vector
	o_extensionsVector.resize(extensionsList.size());
	list<string>::const_iterator itSrc = extensionsList.begin();
	string * itDst = o_extensionsVector.data();
	size_t i = o_extensionsVector.size();
	while ( i-- ) *itDst++ = *itSrc++;
}

bool GlExtensions::has(const string &i_extension)
{
	return binary_search(m_extensions.begin(), m_extensions.end(), i_extension);
}

void GlExtensions::printAll(ostream &io_stream)
{
	io_stream << "vendor  = [" << m_vendor << "]" << endl;
	io_stream << "version = [" << m_version << "]" << endl;
	for (size_t i = 0; i < m_extensions.size(); i++)
		io_stream << setw(3) << setfill('0') << i << ": " << m_extensions[i] << endl;
}

string getGlString(GLenum i_name)
{
	const GLubyte *str = glGetString(i_name);
	return str == NULL ? string() : string((const char *)str);
}

GlExtensions::GlExtensions()
{
	m_vendor = getGlString(GL_VENDOR);
	m_version = getGlString(GL_VERSION);
	parseExtensions(getGlString(GL_EXTENSIONS), m_extensions);
}
