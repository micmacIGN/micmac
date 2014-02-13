#ifndef QT_INTERFACE_ELISE_H
#define QT_INTERFACE_ELISE_H

#include    "StdAfx.h"
#include    "../uti_phgrm/SaisiePts/SaisiePts.h"
#ifdef Int
    #undef Int
#endif
#include    "mainwindow.h"

//using namespace NS_ParamChantierPhotogram;
//using namespace NS_SuperposeImage;

using namespace NS_SaisiePts;

class cQT_Interface  : public cVirtualInterface
{
public :

    cQT_Interface(cAppli_SaisiePts &appli,MainWindow* QTMainWindow);
    ~cQT_Interface();

    void                RedrawAllWindows(){}

    void                Save(){}

    void                SetInvisRef(bool aVal);

    void                DrawZoom(const Pt2dr & aPGlob){} //fenetre zoom

    void                ChangeFreeNamePoint(const std::string &, bool SetFree){}

    void                DeletePoint(cSP_PointGlob *){}

    cCaseNamePoint *    GetIndexNamePoint();

    std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP);



private:    

    void                Init(){}

    MainWindow*         m_QTMainWindow;
};

#endif // QT_INTERFACE_ELISE_H
