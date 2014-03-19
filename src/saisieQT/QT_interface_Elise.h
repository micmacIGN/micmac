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

    void                rebuild2DGlPoints();

    void                rebuild3DGlPoints(cSP_PointeImage* aPIm);

    void                rebuildGlCamera();

    void                option3DPreview();

    void                AddUndo(cOneSaisie * aSom);

    bool                isDisplayed(cImage *aImage);

    void                Redraw(){}

    int                 idPointGlobal(cSP_PointGlob* PG);

    void resizeTable();
    void table_Images_ChangePg(int idPG);
private:

    void                Init();

    MainWindow*         m_QTMainWindow;

    cAppli_SaisiePts*   AppliMetier(){ return  mAppli; }

    Pt2dr               transformation(QPointF pt, int idImage = -1);

    QPointF             transformation(Pt2dr pt, int idImage = -1);

    int                 cImageIdxFromName(QString nameImage);

    int                 cImageIdx(int idGl);

    int                 currentcImageIdx();

    int                 cImageIdxFromGL(cGLData* data);

    cImage *            currentcImage();

    cPoint              selectedPt(int idPt);

    std::string         selectedPtName(int idPt);

    void                addGlPoint(cSP_PointeImage *aPIm, int i);

    cGLData *           getGlData(int idImage);

    cGLData *           getGlData(cImage* image);

    cSP_PointeImage *   currentPointeImage(int idPoint);

    bool                WVisible(cSP_PointeImage &aPIm);

    cData               *_data;

    cCaseNamePoint      *_cNamePt;

signals:

    void                selectPoint(std::string ptName);

    void                dataChanged();

public slots:

    void                rebuildGlPoints(cSP_PointeImage *aPIm = NULL);

    void                changeImages(int idPt, bool aUseCpt);

    void                selectPG(QModelIndex modelIndex);

protected:

    void                rebuild3DGlPoints(cPointGlob *selectPtGlob);

private slots:

    void                addPoint(QPointF point);

    void                movePoint(int idPt);

    void                selectPoint(int idPt);

    void                changeState(int state, int idPt);

    void                removePoint(QString);

    void                setAutoName(QString);

    void                changeName(QString aOldName, QString aNewName);

    void                changeCurPose(void *widgetGL);

    void                filesDropped(const QStringList& filenames, bool setGLData);

    void                SetInvisRef(bool aVal);
};

#endif // QT_INTERFACE_ELISE_H
