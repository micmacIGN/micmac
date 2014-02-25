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

    void                ChangeFreeNamePoint(const std::string &, bool SetFree){}

    cCaseNamePoint *    GetIndexNamePoint();

    std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP);

    void                rebuildGlPoints(cSP_PointeImage *aPIm = NULL);

    void                rebuild2DGlPoints();

    void                rebuild3DGlPoints(cSP_PointeImage* aPIm);

    void                rebuildGlCamera();

    void                option3DPreview();

private:

    void                Init();

    MainWindow*         m_QTMainWindow;

    int                 cImageIdxFromName(QString nameImage);

    int                 cImageIdx(int idGl);

    Pt2dr               transformation(QPointF pt, int idImage = -1);

    QPointF             transformation(Pt2dr pt, int idImage = -1);

    cAppli_SaisiePts*   AppliMetier(){return  mAppli;}

    int                 cImageIdxCurrent();

    std::string         nameSelectPt(int idPt);

    int                 cImageIdxFromGL(cGLData* data);

    void                addGlPoint(cSP_PointeImage *aPIm, int i);

    cGLData *           getGlData(int idImage);

    cSP_PointeImage *   currentPointeImage(int idPoint);

    cImage *            currentCImage();

    cData               *_data;

    bool                WVisible(eEtatPointeImage aState);

    bool                WVisible(cSP_PointeImage &aPIm);

private slots:

    void                addPoint(QPointF point);

    void                movePoint(int idPt);

    void                selectPoint(int idPt);

    void                changeState(int state, int idPt);

    void                changeName(QString aOldName, QString aNewName);

    void                changeCurPose(void *widgetGL);

    void                filesDropped(const QStringList& filenames);

    void                SetInvisRef(bool aVal);
};

#endif // QT_INTERFACE_ELISE_H
