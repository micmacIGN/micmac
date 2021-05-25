/*
drawMask.cpp et drawMask.h regroupe les classes concernant les différentes interface de dessin et visualisation 2D.
Comme il y a beaucoup de fenêtres avec beaucoup d'outils communs, un système d'héritage relie le tout.
PaintInterf et ses sous-classes correspondent à la fenêtre intégrant une barre d'outils de navigation et de dessin ; RenderArea et ses sous-classes correspondent à la zone d'affichage de l'image et des informations graphiques (points homlogues, masque, polygones...) et de tracer.
*/
/*
			RenderArea
			     |	
	         --------------- -------
		|			|
	    DrawArea		VisuHomologues
		|			
	 ----------------------------------------------- 
	|			|			|	
RenderAreaAppui		RenderAreaPlan		RenderAreaSegment	
				|		
			RenderAreaCorrel


			PaintInterf
			     |	
		 -----------------------
		|    			|	
	    DrawInterf		   VueHomologues
		|			
	 ----------------------------------------------- 
	|			|			|	
PaintInterfAppui		PaintInterfPlan		PaintInterfSegment	
				|		
			PaintInterfCorrel

*/
/*
	RenderArea et PaintInterf sont des classes abstraites regroupant les fonctions de visualisation d'une ou plusieurs images et de navigation (zoom in, zoom out, déplacement, plein écran, zoom initial, aide). Il y a un RenderArea par image. PainterScale est l'échelle entre l'image réelle et la taille utile de RenderArea, currentScale est le zoom utilisateur (outils zoom in et zoom out). Tous les segments (VisuHomologues ou DrawArea) sont dessinés dans le référentiel de QPainter pour que les traits aient la même taille apparente.
	VisuHomologues et VueHomologues correspondent au menu Visualisation -> Points homologues ; cette fenêtre permet d'afficher les points homologues de couples d'images (à sélectioner dans des listes) sous-forme de segments par-dessus les images. Seuls les segments dont les deux points sont visibles sont affichés. Les segments sont en réalité constitués de 2 sous-segments (1 par RenderArea) ; pour mettre à jour ces 2 sous-segments à chaque modification d'une RenderArea (avec les outils de navigation), les RenderArea communiquent leur paramètres de position via les fonctions RenderArea::getParamDisplay et PaintInterf::getPosImg.
	DrawArea et DrawInterf sont des classes abstraites regroupant les fonctions de dessin (draw, undo, redo, clear). Les dessins sont enregistrés dans masque sous forme de vecteurs de polygones. Les dessins supprimés sont sauvegardés dans sauvegarde dans le même ordre afin de pouvoir les redessiner. Les compteurs undoCompteur et redoCompteur comptent combien de points sont dessinés en même temps (voir PaintInterfPlan withgradtool) afin de supprimer/redessiner tous ces points en même temps.
	RenderAreaSegment et PaintInterfSegment correspondent à la fois à la saisie de l'axe des abscisses et à la double saisie de l'échelle pour l'orientation "manuelle" (plan horizontal + axe x + échelle) dans la fenêtre Calcul -> Poses. 1 ou 2 RenderArea sont affichées (dans ce cas, le système utilisé par VisuHomologues est repris). C'est un cas particulier de DrawArea : masque ne contient que 1 polygone constitué de maximum 2 points (ou 1 s'il y a 2 RenderArea), et undoCompteur=1.
	PaintInterfPlan et RenderAreaPlan correspondent à la saisie du plan horizontal pour l'orientation "manuelle" ; elles sont aussi les classes mères de RenderAreaCorrel et PaintInterfCorrel ; ces fenêtres permettent la saisie d'un masque. 1 seule RenderArea est affichée. Le masque est constitué d'une suite de polygones ; s'ils sont ouverts, on les dessine sous forme d'une polyligne ; s'ils sont fermés, sous forme d'un polygone non croisé plein. Pour conserver la visibilité de l'image sous-jacente, on utilise l'outil QBrush qui colore 1px/2 (la transparence ne peut être utilisée sinon les polygones qui se superposent somment leur canal alpha et masquent l'image). Il est possible de couper le masque (outil Cut) ; si le polygone n'est pas fermé, il est dessiné sous forme d'une polyligne ; sinon sous forme d'un polygone ayant comme couleur la texture de l'image (grâce à l'outil QBrush). Les segments des polygones peuvent être affinés par l'outil de gradient (d'où l'intérêt des compteurs). Le masque est exporté à la fois sous format tif tuilé (Tiff_Im*) pourle calcul et sous format tif non tuilé pour le réimport. Les masques importés par l'utilisateur sont affichés en transparence (pb d'esthétisme avec le QBrush des polygones, mais le QBrush gère mal le zoom et c'est trop long de refaire un QBrush à chaque zoom).
	PaintInterfCorrel et RenderAreaCorrel correspondent à la saisie du masque pour les cartes de profondeur. Elles ajoutent à la classe RenderAreaPlan un outil de masque automatique à partir des points sift présents dans l'image. Ce masque se construit par regroupement des points proches, puis triangulation des points, éventuellement filtrage des grands triangles pour tenir compte des trous dans le masque, puis délimitation fine du masque par le gradient à partir des triangles.
*/

#ifndef DRAWMASK_H
#define DRAWMASK_H

#include "all.h"

QPointF Pt2dr2QPointF(const Pt2dr& P);
Pt2dr QPointF2Pt2dr(const QPointF& P);
qreal realDistance2(const Pt2dr& P);
qreal realDistance(const Pt2dr& P);

QPoint QPointF2QPoint(const QPointF& P);
QPointF QPoint2QPointF(const QPoint& P);
qreal realDistance2(const QPointF& P);
qreal realDistance(const QPointF& P);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//classes mères pour toutes les interfaces d'affichage (avec possibilité d'afficher N images en même temps => N RenderArea) ; n'est jamais instanciée

class PaintInterf;
class RenderArea : public QWidget
{
	Q_OBJECT

	public:
		enum ToolMode { Draw, Cut, Move, ZoomIn, ZoomOut, Filtre };
		struct ParamDisplay {
			QPoint origine;	//transfoInv(QPoint(0,0))
			qreal globalScale;	//currentScale*painterScale
			qreal painterScale;
			qreal currentScale;
			QPointF center;
			QSize size;	//taille de l'aire de dessin (image->size())
		};

		RenderArea(PaintInterf& parent, const ParamMain& pMain, int N=1, int n=1);
		virtual ~RenderArea();

                #if defined Q_WS_MAC
                        virtual void display(const QString& imageFile, const QList<std::pair<Pt2dr,Pt2dr> >& pts);	//chargement de l'image
                        virtual void display(const QString& imageFile) { display(imageFile, QList<std::pair<Pt2dr,Pt2dr> >()); }
                #else
                        virtual void display(const QString& imageFile, const QList<std::pair<Pt2dr,Pt2dr> >& pts=QList<std::pair<Pt2dr,Pt2dr> >());	//chargement de l'image
                #endif
                virtual void changeSize();			//
		void setToolMode (ToolMode mode);	//modification de l'outil courant
		ToolMode getToolMode () const;	//modification de l'outil courant
		virtual QSize sizeHint ();			//taille initiale
		bool getDone() const;	
		QSize getCurrentSize() const;	
		virtual bool masqueIsEmpty() const;
		virtual bool undoPointsIsEmpty() const;
		virtual bool noPolygone() const;
		ParamDisplay getParamDisplay() const;
		virtual void undoClicked();
		virtual void redoClicked();
		virtual QPoint getPoint() const;
		virtual std::pair<QPoint,QPoint> getSegment() const;
		virtual int getNbPoint() const;
		virtual bool ptSauvegarde() const;
		virtual void setGradTool (bool use);
		virtual void updateGradRegul(int n);
		virtual Tiff_Im* endPolygone ();
		virtual void setPoint(const QPoint& P);

	signals :
		//void sigParam(RenderArea::ParamDisplay,int);	//paramètres d'affichage de l'image (utile pour les segments entre 2 RenderArea)
                void updated(int);	//npos-1

	public slots:
		void zoomFullClicked();			//plein-écran/taille initiale

	protected:
		virtual void paintEvent(QPaintEvent*);		//dessin de l'image
		void resizeEvent(QResizeEvent* event);	//redimensionnement de l'image si resize utilisateur
		virtual void mousePressEvent(QMouseEvent* event);	//navigation
		virtual void mouseMoveEvent(QMouseEvent* event);	//navigation
		virtual void mouseReleaseEvent(QMouseEvent*);	//navigation
		void changeCurrentScale(qreal cScale);	//zoom
		QPointF transfo(QPointF P) const;		//coordonnées souris ->coordonnées image (*image)
		QPointF transfoInv(QPointF P) const;		//coordonnées image (*image) ->coordonnées souris
		QPointF transfo(QPoint P) const;		//coordonnées souris ->coordonnées image (*image)
		QPointF transfoInv(QPoint P) const;		//coordonnées image (*image) ->coordonnées souris
		QRect transfoInv(QRect rect) const;		//coordonnées image (*image) ->coordonnées souris

		QString 		 fichierImage;
		QImage 			 refImage;			// image initiale
		int 			 num;				// si plusieurs images sont affichées, nombre d'images
		int 			 npos;				// si plusieurs images sont affichées, position de l'image (1 à N)
		ToolMode 		 toolMode;			// outil utilisé par paintInterf
		ToolMode 		 oldToolMode;		// sauvegarde du type d'outil de dessin (Draw ou Cut)
		QPointF 		 center;			// centre du zoom dans image
		qreal 			 currentScale; 		// >=1  : échelle du zoom supplémentaire
		qreal 			 painterScale;		// <=1 si l'image est grande  : échelle du painter pour redimensionnement à la taille de la fenêtre
		QPointF 		 dragStartPosition;	// position initiale pour l'outil de déplacement
		bool 			 dragging;			// outil de déplacement en cours d'utilisation
		PaintInterf		*parentWindow;		// accès à la barre d'outil de paintInterf
		QSize 			 currentSize;		// taille de l'image affichée (de image en fait)
		bool 			 done;
		const ParamMain *paramMain;

	private :
};

class PaintInterf : public QDialog
{
	Q_OBJECT

	public:
		PaintInterf(const ParamMain* pMain, Assistant* help, QWidget* parent);
		virtual ~PaintInterf();

		virtual void updateToolBar(const RenderArea::ToolMode& mode);
		RenderArea::ParamDisplay getPosImg(int num) const;	//position d'une image par rapport à l'autre (cas de 2 images uniquement)
		bool getDone() const;

		//virtual void createRenderArea(QStringList* imageFiles);	//création des RenderArea (en fait ne sert pas puisque toujours sous-classée)
		virtual void setLastPoint(int n);
		virtual std::pair<QPoint,QPoint> getSegment() const;
		virtual int getNbPoint(int n) const;
		virtual int maxSpinBox() const;
		virtual Tiff_Im* getMaskImg();
		virtual bool masqueIsEmpty() const;
		QSize sizeHint2() const;

	protected slots:
		virtual void okClicked();
		void dragClicked();
		void zoomInClicked();
		void zoomOutClicked();
		virtual void helpClicked();
		void fullScreenClicked();
		//void getParamImg(RenderArea::ParamDisplay param, int num);
		void updateAll(int pos);

	protected:
		QSize sizeHint() const;
		QSize minimumSizeHint() const;
		QSize maximumSizeHint() const;
		void display();	//ajout des RenderArea à la mise en page
		//void resizeEvent(QResizeEvent*);
		virtual void createActions();
		virtual void createToolBar();


		QList<RenderArea*>  renderArea;
		QVBoxLayout 	   *mainLayout;
		QGroupBox		   *renderBox;
		QAction 		   *okAct;
		QAction 		   *dragAct;
		QAction 		   *zoomInAct;
		QAction 		   *zoomOutAct;
		QAction 		   *zoomFullAct; 
		QAction 		   *helpAct; 
		Assistant		   *assistant;
		QToolBar 		   *toolBar;
		QStatusBar		   *statusBar;
		QPushButton		   *fullScreenButton;

		const ParamMain	   *paramMain;
		QString 			dir;
		QStringList 		imageRef;
		bool 				done;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//affichage des points homologues (cf Conversion->Vue des points homologues)

class VueHomologues;
class VisuHomologues : public RenderArea
{
	Q_OBJECT

	public:
		VisuHomologues(VueHomologues& parent, const ParamMain& pMain, int pos);
		virtual ~VisuHomologues();

		virtual void display(const QString& imageFile, const QList<std::pair<Pt2dr,Pt2dr> >& pts);
		virtual void paintEvent(QPaintEvent* event);

	/*signals :
		void sigParam(RenderArea::ParamDisplay,int);*/	//paramètres d'affichage de l'image (utile pour les segments entre 2 RenderArea)

	private:
		virtual void mousePressEvent(QMouseEvent* event);	//navigation
		virtual void mouseMoveEvent(QMouseEvent* event);	//navigation
		virtual void mouseReleaseEvent(QMouseEvent*);	//navigation

		const QList<std::pair<Pt2dr,Pt2dr> >* points;	//points homologues à tracer
		int num;	//position de l'image (1 ou 2)	
		QRect marquise;
};

class VueHomologues : public PaintInterf
{
	Q_OBJECT

	public:
		VueHomologues(const ParamMain* pMain, Assistant* help, QWidget* parent);
		virtual ~VueHomologues();

		void supprCouples(const QPointF& P, int vue, double rayon);
		void supprCouples(const QRect& R, int vue);
		void supprCouples(const QString& img1, const QString& img2, QList<std::pair<Pt2dr,Pt2dr> >* couples, const QList<QList<std::pair<Pt2dr,Pt2dr> >::iterator>& ASupprimer);

	private slots:
		void liste1Clicked();
		void liste2Clicked();
		virtual void helpClicked();
		void displayFiltreClicked();
		void doMarquiseClicked();
		void undoClicked();
		void redoClicked();
		void saveClicked();

	private:
		struct LiaisCpl {
			LiaisCpl() : image1(QString()), image2(QString()), pointsLiaison(QList<std::pair<Pt2dr,Pt2dr> >()) {}
			LiaisCpl(const QString& img1, const QString& img2) : image1(img1), image2(img2), pointsLiaison(QList<std::pair<Pt2dr,Pt2dr> >()) {}
			~LiaisCpl() {}
			QString image1;
			QString image2;
			QList<std::pair<Pt2dr,Pt2dr> > pointsLiaison;
			bool operator==(const LiaisCpl& lc) { return (image1==lc.image1 && image2==lc.image2); }
		};
		virtual void createActions();
		virtual void createToolBar();
		virtual void updateToolBar(const RenderArea::ToolMode& mode);

		QComboBox* liste1;
		QComboBox* liste2;
		QAction *filtrAct; 
		QToolBar *filtrBar;
		QAction *marquiseAct; 
		QAction *undoAct; 
		QAction *redoAct; 
		QAction *saveAct; 

		QList<LiaisCpl> couples;
		QList<ElSTDNS string> nomFichiers;
		QList<LiaisCpl> undoCouples;
		QList<LiaisCpl> redoCouples;
		bool changed;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//classe mère pour toutes les interfaces de dessin 2D ; n'est jamais instanciée

class DrawInterf;
class DrawArea : public RenderArea
{
	Q_OBJECT

	public:
		class Polygone
		{
			public :
				Polygone();
				Polygone(const ToolMode& t, bool f, const QPolygon& q=QPolygon());
				Polygone(const Polygone& polygone);
				~Polygone();

				Polygone& operator=(const Polygone& polygone);
				Polygone clone() const;	//polygone vide de même outil et de même type

				const ToolMode& getTmode() const;
				bool getFerme() const;
				const QPolygon& getQpolygon() const;

				void setTmode(const ToolMode& t);
				void setFerme(bool f);
				void setQpolygon(const QPolygon& q);
				QPolygon& modifQpolygon();

			private :
				void copie(const Polygone& polygone);

				ToolMode tmode;
				bool ferme;
				QPolygon qpolygon;
		};

	public:
		DrawArea(DrawInterf& parent, const ParamMain& pMain, const QString& imageFile, int N=1, int n=1);
		virtual ~DrawArea();

		virtual bool masqueIsEmpty() const;
		virtual bool undoPointsIsEmpty() const;

	protected:
		virtual void paintEvent(QPaintEvent* event);		//dessin des polygones
		void changeScale(const QList<Polygone>& conteneur, QList<Polygone>& conteneurAlEchelle) const;
		virtual void mousePressEvent(QMouseEvent* event);
		virtual QPolygon* continuePolygone (QList<Polygone>& conteneur, int rad);

		QImage 			refImageClean;	//image de sauvegarde sur laquelle on ne dessine pas le masque précédent du RenderAreaPlan (sert pour le gradient et le Cut)
		QList<Polygone> masque;			//masque en cours de dessin à l'échelle initiale (coordonnées refImage) : le dernier est le plus récent
		QList<Polygone> sauvegarde;		//sauvegarde des dessins supprimés (pour les outils undo-redo)
		int 			updateBuffer;	//buffer pour mettre à jour partiellement le painter autour du dernier dessin
		QList<int> 		undoCompteur;	//sauvegarde du nombre de points dessinés pour l'outil undo : le premier est le plus récent
		QList<int> 		redoCompteur;	//idem pour redo
		bool 			withGradTool;	//outil d'aide à la saisie en cours d'utilisation (false ici)
		int 			maxPoint;		//nombre mawimum de points possibles pour cette image (cas segment)
};


class DrawInterf : public PaintInterf
{
	Q_OBJECT

	public:
		DrawInterf(const ParamMain* pMain, Assistant* help, QWidget* parent);
		virtual ~DrawInterf();

		//virtual void createRenderArea(QStringList* imageFiles);
		virtual void updateToolBar(const RenderArea::ToolMode& mode);
		virtual void createToolBar();

	protected:
		virtual void createActions();

	private slots:
		void drawClicked();
		void cutClicked();

	protected:
		QAction *cutAct;
		QAction *drawAct;
		QAction *undoAct;
		QAction *redoAct;
		QAction *clearAct;

	private:
		int num;	//nombre d'images
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//saisie d'un point

class PaintInterfAppui;
class RenderAreaAppui : public DrawArea
{
	Q_OBJECT

	public:
		RenderAreaAppui(PaintInterfAppui& parent, const ParamMain& pMain, const QString& imageFile, const QPoint& PPrec=QPoint(-1,-1));
		virtual ~RenderAreaAppui();

		virtual QPoint getPoint() const;
		virtual void setPoint(const QPoint& P);

		virtual void undoClicked();
		virtual void redoClicked();

	signals:
                void ptClicked();

	public slots :

	private:
		virtual void mousePressEvent(QMouseEvent * event);
		bool noPoint() const;
};


class PaintInterfAppui : public DrawInterf
{
	Q_OBJECT

	public:
		PaintInterfAppui(const ParamMain* pMain, Assistant* help, const QList<QString>& pointsGPS, const QVector<QVector<QPoint> >& pointsAppui, QWidget* parent);
		virtual ~PaintInterfAppui();

		const QVector<QVector<QPoint> >& getPointsAppui() const;

	public slots:
		void undoClicked();
		void redoClicked();

	private slots:
		virtual void helpClicked();
                void ptClicked();
		void liste1Clicked();
		void liste2Clicked();

	private:
		int getIndexPtApp(int img) const;

		QComboBox* liste1;
		QComboBox* liste2;

		const QList<QString>* ptsGPS;	//points terrain
		QVector<QVector<QPoint> > ptsAppui; //points d'appui (mesures images) : [idxImg][idxPtGPS](ptApp)
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//saisie d'un segment (2 render area)

class PaintInterfSegment;
class RenderAreaSegment : public DrawArea
{
	Q_OBJECT

	public:
		RenderAreaSegment(PaintInterfSegment& parent, const QString& imageFile, const ParamMain& pMain, int N=1, int n=1, const QPoint& P1Prec=QPoint(-1,-1), const QPoint& P2Prec=QPoint(-1,-1));
		virtual ~RenderAreaSegment();

		virtual std::pair<QPoint,QPoint> getSegment() const;
		virtual QPoint getPoint() const;
		virtual int getNbPoint() const;
		virtual bool ptSauvegarde() const;
		virtual void paintEvent(QPaintEvent* event);

		virtual void undoClicked();
		virtual void redoClicked();
		virtual bool noPolygone() const;

	signals:
                void sgtCompleted(bool);

	public slots :
		void clearClicked();

	private:
		virtual void mousePressEvent(QMouseEvent * event);
};



class PaintInterfSegment : public DrawInterf
{
	Q_OBJECT

	public:
		PaintInterfSegment(const ParamMain* pMain, Assistant* help, const std::pair<QString,QString>& images, QWidget* parent, bool planH=true, const QPoint& P1Prec=QPoint(-1,-1), const QPoint& P2Prec=QPoint(-1,-1));
		virtual ~PaintInterfSegment();

		virtual std::pair<QPoint,QPoint> getSegment() const;
		virtual int getNbPoint(int n) const;	//nb de points saisis dans l'autre image
		bool ptSauvegarde(int n) const;	//s'il y a des points sauvegardés dans l'autre image
		virtual void setLastPoint(int n);

	public slots:
		void undoClicked();
		void redoClicked();

	private slots:
		virtual void helpClicked();
                void sgtCompleted(bool completed);	//true : le nb max de points saisissables est atteint pour une image

	private:
		virtual void createActions();

		int lastPoint;	//image du dernier point cliqué
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//saisie d'un masque pour un plan

class PaintInterfPlan;
class RenderAreaPlan : public DrawArea
{
	Q_OBJECT

	public:
		RenderAreaPlan(PaintInterfPlan& parent, const ParamMain& pMain, const QString& imageFile, const QString& masquePrec=QString());
		virtual ~RenderAreaPlan();

		virtual Tiff_Im* endPolygone ();
		virtual void setGradTool (bool use);
		virtual void updateGradRegul(int n);
		virtual QSize sizeHint ();
		virtual bool noPolygone() const;
		virtual void continuePolygone (QList<Polygone>& conteneur, bool upDate);
		QPolygon findGradPath (const QPoint& firstPoint, const QPoint& lastPoint, const QImage& distImg=QImage());

	public slots:
		virtual void undoClicked();
		virtual void redoClicked();
		void clearClicked();

	protected:
		bool loadMasquePrec (const QString& masquePrec);
		virtual void mousePressEvent(QMouseEvent * event);
		void mouseDoubleClickEvent(QMouseEvent* event);
		virtual void paintEvent(QPaintEvent* event);		//dessin du masque à modifier
		void calculeGradient (const qreal& scale, const QRegion& region);
		QPolygon plusCourtChemin (const QPoint& firstPoint, const QPoint& lastPoint, const QRegion& region, const qreal& scale, const QImage& distImg=QImage());
		QPolygon smoothPath (const QPoint& firstPoint, const QPolygon& polyg, int pas, float distance);

		QString					 maskPred;		// masque précédent (ouvert)
		QImage					 masqPrec;		// masque précédent (ouvert)
		std::pair<QImage,QImage> gradient;		// gradient de refImage pour l'aide à la saisie
		float 					 regul;			// coefficient de régularité de l'outil d'aide à la saisie : cout(i) = gradient(i) + regul * dist(i,doite)
		bool					 autoRegul;		// regul calculé automatiquement
		std::pair<QImage,QImage> tempoImages; 	// images rééchantillonnées pour le calcul du gradient
		bool					 refPainted;
};



class PaintInterfPlan : public DrawInterf
{
	Q_OBJECT

	public:
		PaintInterfPlan(const QString& imageFile, const ParamMain* pMain, Assistant* help, QWidget* parent, bool plan=true, const QString& masquePrec=QString(), bool filtre=false);	//plan toujours true sauf si lancé par PaintInterfCorrel
		virtual ~PaintInterfPlan();

		virtual Tiff_Im* getMaskImg();
		virtual int maxSpinBox() const;
		virtual void createToolBar();
		virtual bool masqueIsEmpty() const;

	private slots:
		void gradToolClicked();
		void gradRegulChange(int i);
		virtual void helpClicked();
		virtual void okClicked();

	protected:
		virtual void createActions();

		QAction  *gradToolAct;
		QSpinBox *gradRegulBox;

	private:
		virtual void createRenderArea(const QString& imageFile);

		Tiff_Im *maskImg;
		QString  masqPrec;
		bool	 filtrage;	//activé si interface de dessin de Nuage2Ply (juste pour le titre)
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//saisie d'un masque pour la corrélation (+ masque automatique)


class AutoMask
{
//calcul du masque automatique
	public :

		AutoMask();
		AutoMask(const AutoMask& autoMask);
		AutoMask(const QList<Pt2dr>& liaisons, const QList<pair<Pt3dr, QColor> >& listPt3D, const QImage& refImage, const CamStenope* camera, QList<pair<Pt2dr,QColor> >& ptsLiais);
		~AutoMask ();

		AutoMask& operator=(const AutoMask& autoMask);

		bool isDone() const;
		const QList<QList<Pt2dr> >& getBoitesEnglob() const;
		const QImage& getDistPtLiais() const;
		QImage getDistPtLiais(const QSize& size) const;
		bool isNull() const;

	private :
		void copie(const AutoMask& autoMask);
		bool boiteEnglob (const QList<Pt2dr>& points, QList<Pt2dr>& boite, struct triangulateio&) const;
		void freeTriangulateio(struct triangulateio& triangulation) const;
		bool compareTriangles(const QVector<int>& triangle1, const QVector<int>& triangle2) const;
		bool testTriangles(const QList<Pt2dr>& triangle, double critere) const;
		double aireTriangle(const QList<Pt2dr>& triangle) const;
		double aireBoite(const QList<Pt2dr>& boite) const;

		QList<QList<Pt2dr> > boites;
		QImage distPtLiais;
		bool done;
		bool nullObject;
};


class PaintInterfCorrel;
class RenderAreaCorrel : public RenderAreaPlan
{
	Q_OBJECT

	public:
		RenderAreaCorrel(PaintInterfCorrel& parent, const ParamMain& pMain, const QString& imageFile, const QString& masquePrec=QString());
		virtual ~RenderAreaCorrel();

	public slots:
		void autoClicked(int withHoles);	//0 : false, 1 : true

	private:
		virtual void paintEvent(QPaintEvent* event);		//dessin des points homologues

		AutoMask autoMask;
		QList<pair<Pt2dr,QColor> > ptsLiais; 	//à supprimer
};



class PaintInterfCorrel : public PaintInterfPlan
{
	Q_OBJECT

	public:
		PaintInterfCorrel(const QString& imageFile, const ParamMain* pMain, Assistant* help, QWidget* parent, const QString& masquePrec=QString());
		virtual ~PaintInterfCorrel();

		virtual const QList<Pt2dr>& getPtsLiaison();
		virtual void createToolBar();

	private slots:
		void autoMenuDisplay();
		virtual void helpClicked();

	private:
		virtual void createActions();

		QAction *autoAct;
		QAction *autoFull;
		QAction *autoHoles;

		QList<Pt2dr> pointsLiaison;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class TypeFPRIM
{
	public :
		Pt2dr operator() (const pair<int,Pt2dr> & tq) const;
};
class BenchQdt
{
      public :

         BenchQdt(TypeFPRIM Pt_of_Obj, Box2dr BOX,INT NBOBJMAX,REAL SZMIN);
         bool insert(const pair<int,Pt2dr> p,bool svp=false);
         void remove(const pair<int,Pt2dr>  & p);
	 void clear();
         void voisins(Pt2dr p0,REAL ray, ElSTDNS set<pair<int,Pt2dr> >& S0);

         const Box2dr                    Box;
         const INT                       NbObj;
         const REAL                      SZmin;

      private :
         ElQT<pair<int,Pt2dr>,Pt2dr,TypeFPRIM>      qdt;
};


#endif
