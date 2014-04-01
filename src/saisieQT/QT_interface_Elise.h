#ifndef QT_INTERFACE_ELISE_H
#define QT_INTERFACE_ELISE_H

#include    "mainwindow.h"

class MainWindow;

class cQT_Interface  : public QObject, public cVirtualInterface
{

    Q_OBJECT

public :

    cQT_Interface(cAppli_SaisiePts &appli,MainWindow* QTMainWindow);
    ~cQT_Interface(){}

    void                RedrawAllWindows(){}

    cCaseNamePoint *    GetIndexNamePoint();

    std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP);

    void                AddUndo(cOneSaisie * aSom);

    bool                isDisplayed(cImage *aImage);

    void                Redraw(){}

    cSP_PointGlob *     currentPGlobal() const;

    cImage *            currentCImage();

private:

    //                  init Interface
    void                Init();

    cAppli_SaisiePts*   AppliMetier(){ return  mAppli; }

    //                  Tools cImage                        ///////////////////////////////////////////

    int                 idCImage(QString nameImage);

    int                 idCImage(int idGlWidget);

    int                 idCImage(cGLData* data);

    int                 idCurrentCImage();

    cImage *            CImage(QString nameImage);


    //                  Tools Points                        ///////////////////////////////////////////

    void                removePointGlobal(cSP_PointGlob *pPg);

    int                 idPointGlobal(int idSelectGlPoint);

    QString             namePointGlobal(int idPtGlobal);

    cSP_PointeImage *   PointeImageInCurrentWGL(int idPointGL);

    cSP_PointGlob *     PointGlobInCurrentWGL(int idPointGL);

    Pt2dr               transformation(QPointF pt, int idImage = -1);

    QPointF             transformation(Pt2dr pt, int idImage = -1);

    cSP_PointeImage *   pointeImage(cPointGlob* pg, int idWGL);

    void                centerOnPtGlobal(int idWGL, cPointGlob *aPG);

    void                HighlightPoint(cSP_PointeImage* aPIm);

    void                selectPointGlobal(int idPG);


    virtual eTypePts    PtCreationMode();

    virtual double      PtCreationWindowSize();

    //                  OpenGL                              ///////////////////////////////////////////

    void                addGlPoint(cSP_PointeImage *aPIm, int idImag);

    void                rebuild3DGlPoints(cPointGlob *selectPtGlob);

    void                rebuild2DGlPoints();

    void                rebuild3DGlPoints(cSP_PointeImage* aPIm);

    void                rebuildGlCamera();

    cGLData *           getGlData(int idImage);

    cGLData *           getGlData(cImage* image);

    cPoint              getGLPt_CurWidget(int idPt);

    std::string         getNameGLPt_CurWidget(int idPt);

    ////////////////////////////////////////////////////////////////////////////////////
    //
    cCaseNamePoint      *_cNamePt;

    //                  Fenetre principale
    MainWindow          *m_QTMainWindow;

    cData               *_data;

    int                 _aCpt;

    cSP_PointGlob*      _currentPGlobal;

signals:

    void                dataChanged(cSP_PointeImage *aPIm = NULL);

public slots:

    void                rebuildGlPoints(cSP_PointeImage *aPIm = NULL);

    void                changeImages(int idPt, bool aUseCpt);

    void                selectPointGlobal(QModelIndex modelIndex);

    void                undo(bool);

private slots:

    void                addPoint(QPointF point);

    void                movePoint(int idPt);

    void                selectPoint(int idPtCurGLW);

    void                changeState(int state, int idPt);

    void                removePoint(QString);

    void                setAutoName(QString);

    void                changeName(QString aOldName, QString aNewName);

    void                changeCurPose(void *widgetGL);

    void                filesDropped(const QStringList& filenames);

    void                SetInvisRef(bool aVal);

    void                close();
};

#endif // QT_INTERFACE_ELISE_H
