#include "Engine2D.h"

cLoader::cLoader()
 : m_FilenamesIn(),
   m_FilenamesOut()
{}

void cLoader::SetFilenamesOut()
{
    m_FilenamesOut.clear();

    for (int aK=0;aK < m_FilenamesIn.size();++aK)
    {
        QFileInfo fi(m_FilenamesIn[aK]);

        m_FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_masq.tif");
    }
}

//****************************************
//   cEngine

cEngine2D::cEngine2D():
    m_Loader(new cLoader)
{}

cEngine2D::~cEngine2D()
{
   delete m_Loader;
}
