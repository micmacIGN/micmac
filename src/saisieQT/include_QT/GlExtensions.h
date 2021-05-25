#ifndef __GL_EXTENSIONS_INFO__
#define __GL_EXTENSIONS_INFO__

#include <string>
#include <vector>
#include <iostream>

class GlExtensions
{
private:
	std::string m_vendor, m_version;
	std::vector<std::string> m_extensions;

public:
	GlExtensions();

	bool has(const std::string &i_extension);

	void printAll(std::ostream &aStream = std::cout);

	const std::string & vendor() const { return m_vendor; }
	const std::string & version() const { return m_version; }
};

std::string getGlString(const char *i_str);

void parseExtensions(const std::string &i_extensionsString, std::vector<std::string> &o_extensionsVector);

#endif
