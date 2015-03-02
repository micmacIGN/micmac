#ifndef QT_INTERFACE_ELISE_H
#define QT_INTERFACE_ELISE_H

#include "saisieQT_window.h"

class SaisieQtWindow;

class cQT_Interface  : public QObject, public cVirtualInterface
{

    Q_OBJECT

public :

    cQT_Interface(cAppli_SaisiePts &appli,SaisieQtWindow* QTMainWindow);
    ~cQT_Interface(){}

    void                RedrawAllWindows(){}

    cCaseNamePoint *    GetIndexNamePoint();

    std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP);

    void                AddUndo(cOneSaisie * aSom);

    bool                isDisplayed(cImage *aImage);

    void                Redraw(){}

    cSP_PointGlob *     currentPGlobal() const;

    cImage *            currentCImage();

    cData *             getData() { return _data; }

    int                 getQTWinMode();

    void                Warning(std::string aMsg);

    int                 idCImage(QString nameImage);

private:

    //                  init Interface
    void                Init();

    cAppli_SaisiePts*   AppliMetier(){ return  mAppli; }

    //                  Tools cImage                        ///////////////////////////////////////////

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
    SaisieQtWindow      *m_QTMainWindow;

    cData               *_data;

    int                 _aCpt;

    void                setCurrentPGlobal(cSP_PointGlob* pg){_currentPGlobal = pg;}

    cSP_PointGlob*      _currentPGlobal;

    QMenu               *_menuPGView;
    QMenu               *_menuImagesView;

    QAction             *_thisPointAction;
    QAction             *_thisImagesAction;

    QSignalMapper       *_signalMapperPG;

    QStringList         _listSelectedPG;

signals:

    void                dataChanged(bool save = false, cSP_PointeImage *aPIm = NULL);

public slots:

    void                cmdBascule();

    void                rebuildGlPoints(bool bSave = false, cSP_PointeImage *aPIm = NULL);

    void                changeImages(int idPtGl, bool aUseCpt);

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

    void                contextMenu_PGsTable(const QPoint &widgetXY);

    void                changeImagesPG(int idPg, bool aUseCpt = false);

    void                contextMenu_ImagesTable(const QPoint &widgetXY);

    void                viewSelectImages();

    void                deleteSelectedGlobalPoints();

    void                validateSelectedGlobalPoints();
};



class cCamHandlerElise : cCamHandler
{
public:
	cCamHandlerElise(CamStenope *pCam) :
		_Cam(pCam){

	}

	virtual void getCoins(Pt3dr &aP1,Pt3dr &aP2,Pt3dr &aP3,Pt3dr &aP4, double aZ)
	{
		_Cam->Coins(aP1, aP2, aP3, aP4, aZ);
	}

	virtual Pt3dr getCenter()
	{
		return _Cam->VraiOpticalCenter();
	}

private:

	CamStenope *_Cam;
};

class deviceIOCameraElise : deviceIOCamera
{


public:

	deviceIOCameraElise():_mnICNM(NULL){}

	virtual cCamHandler*  loadCamera(QString aNameFile)
	{
		QFileInfo fi(aNameFile);

		if(_mnICNM == NULL || _oldPathChantier != fi.dir())
		{
			_oldPathChantier = fi.dir();
			string DirChantier = (fi.dir().absolutePath()+ QDir::separator()).toStdString();
			_mnICNM = cInterfChantierNameManipulateur::BasicAlloc(DirChantier);

			DUMP_LINE

		}

		cCamHandlerElise *camElise = new cCamHandlerElise(CamOrientGenFromFile(fi.fileName().toStdString(),_mnICNM, false));

		// TODO delete 		anICNM???


		return (cCamHandler*) camElise;
	}

private:

	QDir							  _oldPathChantier;

	cInterfChantierNameManipulateur * _mnICNM;
};

#endif // QT_INTERFACE_ELISE_H
