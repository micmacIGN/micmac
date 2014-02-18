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

class MainWindow;

class cQT_Interface  : public QObject, public cVirtualInterface
{

    Q_OBJECT

public :

    cQT_Interface(cAppli_SaisiePts &appli,MainWindow* QTMainWindow);
    ~cQT_Interface(){}

    void                RedrawAllWindows(){}

    void                Save(){}

    void                SetInvisRef(bool aVal);

    void                DrawZoom(const Pt2dr & aPGlob){} //fenetre zoom

    void                ChangeFreeNamePoint(const std::string &, bool SetFree){}

    void                DeletePoint(cSP_PointGlob *){}

    cCaseNamePoint *    GetIndexNamePoint();

    std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP);

    cAppli_SaisiePts*   AppliMetier(){return  mAppli;}

    void                rebuildGlPoints();

    int                 cImageIdxCurrent();

    std::string         nameSelectPt(int idPt);

    int                 cImageIdxFromGL(cGLData* data);

    void                addGlPoint(const cOneSaisie& aSom, int i);

    cGLData *           getGlData(int idImage);

    cSP_PointeImage * currentPointeImage(int idx);
    cImage * currentCImage();
private:

    void                Init(){}

    MainWindow*         m_QTMainWindow;

    int                 cImageIdxFromName(QString nameImage);

    int                 cImageIdx(int idGl);

    Pt2dr               transformation(QPointF pt, int idImage = -1);

    QPointF             transformation(Pt2dr pt, int idImage = -1);

private slots:

    void                addPoint(QPointF point);

    void                movePoint(int idPt);

    void                changeState(int state, int idPt);
};

#endif // QT_INTERFACE_ELISE_H
