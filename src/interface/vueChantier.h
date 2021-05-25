/*
vueChantier.h et vueChantier.cpp correspondent aux fenêtres Visualisation -> Vue du chantier et Visualisation -> Vue des nuages de points.
Cette fenêtre permet d'afficher les paramètres du chantiers (position et orientation des caméras, points homologues en 3D, emprise des images et éventuellement nuages de points issues de MICMAC) et de naviguer en 3D.
La fenêtre est définie par la classe VueChantier. La zone de navigation 3D correspond à la classe GLWidget ; elle utilise openGL.
Dans la barre d'outils, l'outil de rotation est défini par la classe RotationButton, les liste des objets à afficher par Layers (une liste pour les caméras, les points homologues et les emprises, une autre pour les différents nuages), la partie permettant de saisir l'image de référence ou les images pour la corrélation (pour la partie MICMAC) par SelectCamBox.
La classe GLParams regroupe les paramètres des objets à afficher ainsi que les paramètres d'affichage de GLWidget (point de vue...). NB : QGLWidget a un bug : quand la fenêtre est réduite ou cachée, QGLWidget perd ses paramètres d'affichage et se fige ; la solution développée ici est de supprimer GLWidget et d'en créer un nouveau en la paramétrant avec les anciens paramètres d'affichage afin que l'utilisateur ne le voie pas ; c'est pourquoi les paramètres d'affichage sont sauvegardés hors de GLWidget, dans GLParams.
La classe Nuage regroupe les paramètres de chaque nuage à afficher ; un nuage est composé de 6 nuages identiques à différentes résolutions ; ils sont tous créés au départ et seulement affichés/cachés par l'utilisateur afin de fluidifier l'affichage.
La classe Pose regroupe les paramètres relatifs à l'orientation d'une image ; elle est aussi utilisée dans d'autres fichiers
*/

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "all.h"
 
#define GLWidget GLWidget_IC

class Pose
	//paramètres de pose d'une caméra
{
	public :
		Pose();
		Pose(const Pose& pose);
		~Pose();

               Pose& operator=(const Pose& pose);
	
		void setNomImg(const QString& img);
		void setCamera(CamStenope* cam);
		QList<pair<Pt3dr, QColor> >& modifPtsAppui();
		QList<pair<Pt3dr, QColor> >& modifPtsAppui2nd();
		void setEmprise(const QVector<Pt3dr>& rec);
		void setImgSize(const QSize& s);

		const QString& getNomImg() const;
		const CamStenope& getCamera() const;
		const ElMatrix<REAL>& rotation() const;	
		const QList<pair<Pt3dr, QColor> >& getPtsAppui() const;
		const QList<pair<Pt3dr, QColor> >& getPtsAppui2nd() const;
		const QVector<Pt3dr>& getEmprise() const;
                Pt3dr centre() const;
		int width() const;
		int height() const;

		QVector<REAL> centre2() const;	//REAL[3], = centre()
		QVector<REAL> direction() const;
		void simplifie2nd();

	private :
		void copie(const Pose& pose);

		QString						nomImg;
		const CamStenope				   *camera;
		QList<pair<Pt3dr, QColor> > ptsAppui;
		QList<pair<Pt3dr, QColor> > ptsAppui2nd;	//points d'appui qui ne seront tracés que si on cherche tous les points de cette caméra (infoButton)
		QVector<Pt3dr>				emprise;
		QSize						imgSize;
};

class SelectCamBox : public QGroupBox
//pour micmac (paramètres d'un masque) : permet de sélectionner l'image de référence ou les images utilisées pour la corrélation (ou pas d'affichage de la QGroupBox)
{
	Q_OBJECT

	public:
		enum Mode { Hide, RefImage, CorrelImages };
		
		SelectCamBox();
		~SelectCamBox();

		void create(const Mode& mode, const QString& refImg=QString(), const QStringList& precCam=QStringList());
		void setRefButtonChecked(bool b);

                QStringList getSelectedCam() const;
		const Mode& getMode() const;
		bool isRefButtonChecked() const;

	public slots :
		void addCamera(const QString& camera);

	signals :
		void refClicked();
		void okClicked();

	private slots:
		void cut();

	private :
		void clearContent();
		void contextMenuEvent(QContextMenuEvent *event);

		QHBoxLayout	*mainLayout;
		QToolButton *refButton;
		QToolButton *okButton;
		QListWidget *camList;
		QAction		*cutAct;
		QLineEdit	*camEdit;

		QString refImage;
		Mode 	pCurrentMode;		
};

class GeorefMNT
//géoréférencement des cartes de profondeur, TA et orthoimages mosaïquées (fichiers Z_NumX_DeZoomY_GeomIm-N.xml)
{
	public :
		GeorefMNT();
		GeorefMNT(const QString& f, bool lire=false);
		GeorefMNT(const GeorefMNT& georefMNT);
		~GeorefMNT();

		GeorefMNT& operator=(const GeorefMNT& georefMNT);

		Pt2dr terrain2Image(const Pt2dr& P) const;

		bool getDone() const;
		double getX0() const;
		double getY0() const;
		double getDx() const;
		double getDy() const;
		const QString& getFichier() const;
		bool getGeomTerrain() const;

		void setX0(double x);
		void setY0(double y);
		void setDx(double d);
		void setDy(double d);
		void setFichier(const QString& f);
		void setGeomTerrain(bool g);

	private :
		void copie(const GeorefMNT& georefMNT);

		double	x0;
		double	y0;
		double	dx;
		double	dy;
		QString fichier;
		bool	geomTerrain;
		bool	done;
};

class Nuage {
	public :
		Nuage();
		Nuage(const Nuage& nuage);
		~Nuage();
		
		Nuage& operator=(const Nuage& nuage);

		const QVector<cElNuage3DMaille*>& getPoints() const;
		const QVector<QString>& getCorrelation() const;
		const QString& getCarte() const;
		const QString& getImageCouleur() const;
		const Pose& getPose() const;
		int getFocale() const;
		const QVector<REAL>& getZone() const;
		int getNbResol() const;
		bool getFromTA() const;
		const GeorefMNT& getGeorefMNT() const;
		int getZoomMax() const;

                QVector<cElNuage3DMaille*>& modifPoints();
		QVector<QString>& modifCorrelation();
		void setCarte(const QString& c);
		void setImageCouleur(const QString& ic);
                void setPose(Pose& p);
                Pose& modifPose();
		void setFocale(int f);
                void setZone(const QVector<REAL>& z);
		void setFromTA(bool b);
		void setGeorefMNT(const GeorefMNT& grm);
		void setZoomMax(int z);

	private :
		void copie(const Nuage& nuage);

		QVector<cElNuage3DMaille*> points;	//0 : peu précis, 5 : resolution 1
		QVector<QString> correlation;	//images de corrélation correspondantes (indique la fiabilité de la corrélation)
		QString carte;	//avec dossier !
		QString imageCouleur;
                Pose* pose; //pointeur vers GLParams.poses.at(?)
		int focale;
		QVector<REAL> zone;
		bool fromTA;
		static int nbResol;
		GeorefMNT georefMNT;
		int zoomMax;
};


class GLParams
{
	public :
		enum Couleur {Mono, Hypso, Texture};
		enum Mesure {Aucune, S1, P1, V1, S2, P2, V2};	//mesure de distance : sélection+clic point1+validation puis idem point2

		GLParams();
		GLParams(const GLParams& params);
		~GLParams();

		GLParams& operator=(const GLParams& params);

                bool addNuages(const ParamMain& paramMain);

		const QString& getDossier() const;
		const QVector<Pose>& getPoses() const;
		double getDistance() const;
		const QVector<GLdouble>& getZoneChantier() const;
		const QVector<GLdouble>& getZoneChantierEtCam() const;
		const GLdouble* getRot() const;
		const QVector<GLdouble>& getTrans() const;
		const GLdouble& getScale() const;
		int getFocale() const;
		const QVector<bool>& getCamlayers() const;
		const QVector<bool>& getNuaglayers() const;
		const QVector<Nuage>& getNuages() const;
		const QVector<REAL>& getZonenuage() const;
		const Couleur& getColor() const;
		const Mesure& getMesure() const;
                const std::pair<Pt3dr,Pt3dr>& getSegment() const;
                GLuint getNbGLLists() const;
		double getMaxScale() const;

		void setDossier(const QString& dir);
		QVector<Pose>& modifPoses();
		void setDistance(double echelle);
		void setZoneChantier(const QVector<GLdouble>& zc);
		void setZoneChantierEtCam(const QVector<GLdouble>& zcec);
		GLdouble*& modifRot();
		QVector<GLdouble>& modifTrans();
		void setScale(const GLdouble& sc);
		void setFocale(int f);
		QVector<bool>& modifCamlayers();
		QVector<bool>& modifNuaglayers();
		QVector<Nuage>& modifNuages();
		QVector<REAL>& modifZonenuage();
		void setColor(const Couleur& c);
		void setMesure(const Mesure& m);
		void setSegment(const std::pair<Pt3dr,Pt3dr>& p);
		void incrNbGLLists();
		void resetNbGLLists();

	private :
		void copie(const GLParams& params);

		QString dossier;
                QVector<Pose> poses;   //n = imgToOri.count()
		double distance;
		QVector<GLdouble> zoneChantier;	//xmin, xmax, ymin, ymax, zmin, zmax
		QVector<GLdouble> zoneChantierEtCam;	//xmin, xmax, ymin, ymax, zmin, zmax
		GLdouble* rot;
		QVector<GLdouble> trans;
		GLdouble scale;
		int focale;
		QVector<bool> camlayers;
		QVector<bool> nuaglayers;
                QVector<Nuage> nuages;  //n = masques.count()
		QVector<REAL> zoneNuage;
		Couleur color;
		Mesure measure;
		std::pair<Pt3dr,Pt3dr> sgt;
		static GLuint nbGLLists;
		static double maxScale;
};

class GLWidget : public QGLWidget
{
	Q_OBJECT

	public:
		GLWidget(VueChantier *parent, GLParams* params, const ParamMain& pMain);
		~GLWidget();

		void setInfo(bool b);
		void setRef(bool b);
                void setColor(const GLParams::Couleur& couleur);
                void setMesure(const GLParams::Mesure& measure);
		GLuint makeObjectNuag(int num, int etape);

                int getRefImg() const;

	signals :
		void changeFocale(int);
                void cameraSelected(const QString&);

	public slots:
		//outils
    		void reinit();
		void translate(int direction);
		void rescale(int directioOrientationConiquen);
		void modifFocale(int value);
		void rotate(int rotation, double angle);
		void dispCamLayers(int layer, bool display);
		void dispNuagLayers(int layer, bool display);
		void addNuages();

	protected:
		void initializeGL();
		void paintEvent(QPaintEvent*);
		void resizeGL(int width, int height);
		//clavier
		void keyPressEvent(QKeyEvent * event);
		//souris
		void mousePressEvent(QMouseEvent *event);
		void mouseMoveEvent(QMouseEvent *event);
		void mouseReleaseEvent(QMouseEvent *event);
    		void wheelEvent(QWheelEvent *event);

	private slots:
		void addCam();

	private:
		QSize sizeHint() const;
		GLuint makeObjectCam();
		GLuint makeObjectEmp();
		GLuint makeObjectApp();
		GLuint makeAxes();
		GLuint makeBoule();
		void glCircle3i(GLint radius, GLdouble * m);
		bool calcDezoom(int num);
		void setupViewport(int width, int height);
		void setRotation(GLdouble* R);
                void setTranslation(const QVector<GLdouble>& T);
                void setScale(const GLdouble& sc);
                void convertTranslation(int direction, const GLdouble& T);
                void convertRotation(int direction, const GLdouble& R, bool anti);
		void multScale(int diff);
                pair<QVector<GLdouble>,QVector<GLdouble> > getMouseDirection (const QPoint& P, GLdouble * matrice) const;
                QVector<GLdouble> getSpherePoint(const QPoint& P) const;
                QVector<GLdouble> getPlanPoint(const pair<QVector<GLdouble>,QVector<GLdouble> >& direction) const;
                GLdouble profondeur(const Pt3dr& point) const;
		void doNuages();

		GLParams		  *parametres;
		QVector<GLdouble>  espace;			//xmin, xmax, ymin, ymax, zmin, zmax
		QVector<int>	   currentDezoom;	//nuages->count()] , de 0 (peu précis,  dézoom 32) à 5 (résolution 1)
		const ParamMain	  *paramMain;

		GLuint			  objectCam;
		GLuint			  objectEmp;
		GLuint			  objectApp;
		QVector<GLuint>   objectNuag;	//[6*num + resol], du moins résolu (1/32) au plus résolu (1)
		GLuint 			  axes;
		GLuint			  boule;
		QPoint 			  lastPos;
		pair<int,double>  posSphere;
		double			  winZ;
		bool			  info;
		bool			  ref;
		QPair<QPoint,int> infoBulle;
		double			  demiCote;
		QAction			 *addCamAct;
		static double 	  visibilite;	//taille de l'espace / taille du chantier
};

class Layers;
class VueChantier : public QDialog
//visualise le chantier en 3D
{
	Q_OBJECT

	public:
		VueChantier(const ParamMain* pMain, QWidget* parent, Assistant* help);
		~VueChantier();

		const QStringList getRefImg() const;
		const QVector<Pose>& getPoses() const;
                const GLParams& getParams() const;
                bool isDone() const;

                void show(const SelectCamBox::Mode& refMode, const QString& refImg=QString(), const QStringList& precCam=QStringList());
                bool addNuages(const QVector<QString>& imgRef, const ParamMain& paramMain);
                GLParams& modifParams();

                static QString getHomol3D (const QString& nomPose, const QString& dossier, QList<pair<Pt3dr, QColor> >& listPt3D);
                static QString convert (const ParamMain* pMain, QVector<Pose>& cameras, int N=-1);

	signals :
		void layChecked(int,bool);
		void dispNuages();

	private slots:
		void answerButton(int button=-1);
		void infoClicked();
		void hypsoClicked();
		void textureClicked();
		void noColorClicked();
		void measureClicked();
		void refClicked();
		void okClicked();
		void helpClicked();
		void setFocale(int f);

	protected :

	private:
		QSize sizeHint();
		QSize minimumSizeHint();
		QSize maximumSizeHint();
		virtual bool getParamPoses();
		void initialise();
		void hideEvent(QHideEvent*);
		void showEvent(QShowEvent*);
		void resizeEvent(QResizeEvent*);

		GLWidget 		*glWidget;
		QToolBar		*toolBar1;
		QToolBar		*toolBar2;
		QGroupBox		*translationBox;
		QGroupBox		*zoomBox;
		QGroupBox		*focaleBox;
		QGroupBox		*rotationBox;
		QToolButton	   **moveButtons;
		QToolButton		*intiViewButton;
		QSlider			*focaleSlide;
		RotationButton	*rotationButton;
		Layers			*aperoLayers;
		Layers			*nuagesLayers;
		QToolButton		*infoButton;
		QToolButton		*colorButton;
		QToolButton		*measureButton;
		SelectCamBox	*refBox;
		QToolButton		 *helpButton;
		QTimer			*timer;
		int				 currentTool;
		QSignalMapper	*mapper3;
		QGridLayout		*mainLayout;
		Assistant		*assistant;
		bool			 hidden;
		bool			 done;
		const ParamMain	*paramMain;
		GLParams		 glparams;
};

class RotationButton : public QToolButton
//bouton de rotation dynamique
{
	Q_OBJECT

	public:
		RotationButton();
		~RotationButton();
		int getRotation();

	signals :
		void roll(int rot, double angle);

	private :
		void mousePressEvent(QMouseEvent * event);
		void mouseMoveEvent(QMouseEvent * event);
		void mouseReleaseEvent(QMouseEvent * event);

		int rotation;
		QPoint lastPos;
		QPoint pressPos;
		QIcon* icone;
		bool rot0;
};

class Layers : public QGroupBox
//permet de gérer les différents objets à afficher/masquer
{
	Q_OBJECT

	public:
		Layers();
		~Layers();

		void create(QString title, const QVector<QString>& noms);
		void setChecked(int layer, bool checked);
		void addWidget(QWidget* widget, Qt::Alignment alignment);

	signals :
		void stateChanged(int layer, bool isChecked);

	private slots:
			void emitChanges(int layer);

	private :
		QCheckBox** ckeckBoxes;
		int count;
		QVBoxLayout* layout;
};

#endif
