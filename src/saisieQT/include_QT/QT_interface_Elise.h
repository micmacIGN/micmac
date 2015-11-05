#ifndef QT_INTERFACE_ELISE_H
#define QT_INTERFACE_ELISE_H

#include "StdAfx.h"
#include "Elise_QT.h"
#include "saisieQT_window.h"
#include  "Tree.h"

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

	static void			toQVec3D(Pt3dr P,QVector3D &qP);

	static QVector3D	toQVec3D(Pt3dr P);

	static void			connectDeviceElise(SaisieQtWindow& win);

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

protected:

	bool				isPolygonZero();

	float				lengthRule();
protected slots:

	void				updateToolBar();

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

	virtual void getCoins(QVector3D &aP1,QVector3D &aP2,QVector3D &aP3,QVector3D &aP4, double aZ)
	{
		Pt3dr P1;
		Pt3dr P2;
		Pt3dr P3;
		Pt3dr P4;

		_Cam->Coins(P1, P2, P3, P4, aZ);

		cQT_Interface::toQVec3D(P1,aP1);
		cQT_Interface::toQVec3D(P2,aP2);
		cQT_Interface::toQVec3D(P3,aP3);
		cQT_Interface::toQVec3D(P4,aP4);

	}

	virtual QVector3D getCenter()
	{
		return cQT_Interface::toQVec3D(_Cam->VraiOpticalCenter());
	}

	virtual QVector3D getRotation()
	{
		ElRotation3D orient = _Cam->Orient();

		if(orient.IsTrueRot())
			return QVector3D(orient.teta01(),orient.teta02(),orient.teta12());
		else
			return QVector3D();
	}

private:

	CamStenope *_Cam;
};

class deviceIOTieFileElise : public deviceIOTieFile
{
public:
	virtual void  load(QString aNameFile,QPolygonF& poly)
	{
		std::vector<DigeoPoint> vDigPt;
		DigeoPoint::readDigeoFile(aNameFile.toStdString(),false,vDigPt);
		for (int i = 0;  i < (int)vDigPt.size(); ++i)
		{
			DigeoPoint& d = vDigPt[i];
			poly.push_back(QPointF(d.x,d.y));
		}
	}
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

		}

		cCamHandlerElise *camElise = new cCamHandlerElise(CamOrientGenFromFile(fi.fileName().toStdString(),_mnICNM, false));

		// TODO delete 		anICNM???


		return (cCamHandler*) camElise;
	}

private:

	QDir							  _oldPathChantier;

	cInterfChantierNameManipulateur * _mnICNM;
};

class deviceIOImageElise : public deviceIOImage
{
public:
	virtual QImage*	loadImage(QString aNameFile,bool OPENGL = true)
	{
		Tiff_Im aTF= Tiff_Im::StdConvGen(aNameFile.toStdString(),3,false);

		Pt2di aSz = aTF.sz();

		//maskedImg->_m_image = new QImage(aSz.x, aSz.y, QImage::Format_RGB888);
		QImage tempImageElIse(aSz.x, aSz.y,QImage::Format_RGB888);

		Im2D_U_INT1  aImR(aSz.x,aSz.y);
		Im2D_U_INT1  aImG(aSz.x,aSz.y);
		Im2D_U_INT1  aImB(aSz.x,aSz.y);

		ELISE_COPY
		(
		   aTF.all_pts(),
		   aTF.in(),
		   Virgule(aImR.out(),aImG.out(),aImB.out())
		);

		U_INT1 ** aDataR = aImR.data();
		U_INT1 ** aDataG = aImG.data();
		U_INT1 ** aDataB = aImB.data();

		for (int y=0; y<aSz.y; y++)
		{
			for (int x=0; x<aSz.x; x++)
			{
				QColor col(aDataR[y][x],aDataG[y][x],aDataB[y][x]);

				tempImageElIse.setPixel(x,y,col.rgb());
			}
		}

		if(OPENGL)
			return new QImage(QGLWidget::convertToGLFormat( tempImageElIse ));
		else
			return new QImage(tempImageElIse);
	}

	virtual QImage*	loadMask(QString aNameFile)
	{
		Tiff_Im imgMask( aNameFile.toStdString().c_str() );

		if( imgMask.can_elise_use() )
		{
			int w = imgMask.sz().x;
			int h = imgMask.sz().y;

			QImage tempMask ( w, h, QImage::Format_Mono);
			tempMask.fill(0);

			Im2D_Bits<1> aOut(w,h,1);
			ELISE_COPY(imgMask.all_pts(),imgMask.in(),aOut.out());

			for (int x=0;x< w;++x)
				for (int y=0; y<h;++y)
					if (aOut.get(x,y) == 1 )
						tempMask.setPixel(x,y,1);

			return new QImage(QGLWidget::convertToGLFormat( tempMask ));
		}
		else return NULL;
	}

	virtual void	doMaskImage(QImage &mask,QString &aNameFile)
	{
		cFileOriMnt anOri;

		anOri.NameFileMnt()		= aNameFile.toStdString();
		anOri.NombrePixels()	= Pt2di(mask.width(),mask.height());
		anOri.OriginePlani()	= Pt2dr(0,0);
		anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
		anOri.OrigineAlti()		= 0.0;
		anOri.ResolutionAlti()	= 1.0;
		anOri.Geometrie()		= eGeomMNTFaisceauIm1PrCh_Px1D;

		MakeFileXML(anOri, StdPrefix(aNameFile.toStdString()) + ".xml");
	}
};

#endif // QT_INTERFACE_ELISE_H
