/*
interfApero.h et interfApero.cpp correspondent au menu Calcul -> Poses qui permet de calculer les poses des caméras ainsique la calibration interne de chaque objectif.
La classe InterfApero correspond à cette fenêtre.
Elle est composée d'onglets et permet:
- de sélectionner les images à orienter -> onglet ImgToOriTabA ;
- de sélectionner l'image maîtresse -> onglet MaitresseTabA;
- de définir l'orientation absolue (orientation relative à l'image maîtresse - "en l'air" -, orientation selon un plan, une direction et une échelle - orientation "manuelle" - ou géoréférencement par points d'appui - non programmé) -> onglet ReferenceTabA :
	* la QGroupBox manuBox correspond à la partie orientation manuelle, MasqueWidget masqueWidget correspond à la partie saisie d'un plan horizontale (elle est définie dans interfaMicmac.h), les PaintInterfSegment paintInterfSegment correspondent aux interfaces de saisie de segment dans une image (soit pour la direction, soit pour l'échelle; elle est définie dans drawMask.h).
- de précéder le calcul d'une auto-calibration, en sélectionant les images pour ce pré-calcul -> onglet AutoCalibTabA;
- d'effectuer le calcul par blocs d'images de même focale, en sélectionnant les calibrations dont les poses des images correspondantes sont à calculer en premier -> onglet MultiEchelleTabA ;
- d'ajouter une phase de calcul où les images ont des calibrations différentes, en sélectionnant les calibrations à dissocier -> onglet LiberCalibTabA ;
- de préfiltrer les points homologues pour diminuer leur densité ou de ne pas les affciher dans la vue 3D -> onglet PtsHomolTabA.
La classe ParamApero regroupe tous les paramètres saisis.
La classe UserOrientation regroupe les paramètres saisis pour l'orientation absolue.
*/

#ifndef INTERFAPERO_H
#define INTERFAPERO_H

#include "all.h"


class InterfApero;
class ParamApero;


class MasqueWidget : public QWidget
//classe qui regroupe les GroupBox utiles pour saisir le masque (utilisées par interfMicmac->maskTab, interfMicmac->repereTab et par interfApero->referenceTabA)
{
	Q_OBJECT

	public:
		MasqueWidget(const ParamMain* param, Assistant* help, bool mm=false, bool mmasq=false, QPushButton* vue3DButton=0, const QString& image=QString(), const QString& postfx=QString("_MasqPlan"));	//si TA, image=imageFond=chemin complet
		~MasqueWidget();

		enum Mode { Begin, Image, NewMasque, OpenMasque, Enreg };

		QGroupBox* getMasqueBox();
		const Mode& getCurrentMode() const;
		void setImageFond(const QString& img);	//nom sans le chemin
		static QString convert2Rgba(const QString& tuiledFile, bool toMask, const QString& newFile=QString());
		void updateParam(ParamApero* parametres);

	signals:
		void updateParam();

	public slots:
		void imgsSetChanged();
		void updateParam(CarteDeProfondeur* parametres, bool repere);

	private slots:
		void choiceClicked();
		void comboChanged(QString txt);
		void modifClicked();
		void openClicked();

	private:
		bool calcMasqueFile();
		void updateInterface(Mode mode);
		void showPainter(QString masquePrec=QString());
		void saveClicked();

		QGroupBox	 *imageBox;
		QComboBox	 *imageCombo;
		QGroupBox 	 *radioBox;
		QRadioButton *radioNvu;
		QRadioButton *radioOpen;
		QGroupBox	 *openBox;
		QLineEdit	 *openEdit;
		QPushButton	 *openButton;
		QPushButton	 *modifButton;
		QGroupBox	 *saveBox;
		QLineEdit	 *saveEdit;
		QGroupBox	 *masqueBox;

		//données
		Mode 	 currentMode;
		Tiff_Im *masque;

		Assistant		*assistant;
		const ParamMain *paramMain;
		QString 		 dir;		
		PaintInterf		*paintInterf;
		QString			 imageFond;
		QString			 masqueFile;
		QString			 postfixe;	//cas repere apero absolu : "_MasqPlan", cas repereTA micmac : "_MasqRepTA", cas micmac : "_masque"
		bool			 micmac;
		bool			 micmacMasque;
};

class DirectionWidget : public QWidget
{
	Q_OBJECT

	public :
		DirectionWidget(const ParamMain* pMain, const QStringList & liste, Assistant* help, const std::pair<QString,QString>& imgPrec/*=std::pair<QString,QString>()*/, int N/*=2*/, const std::pair<QPoint,QPoint>& ptsPrec/*=std::pair<QPoint,QPoint>(QPoint(-1,-1),QPoint(-1,-1))*/, const QPoint& axePrec=QPoint(1,0));
		~DirectionWidget();

		QGroupBox* getBox();
		void updateParam(ParamApero* parametres);

	signals:
		void updateParam();

	public slots :
		void updateParam(CarteDeProfondeur* parametres);

	private slots :
		void imageDirClicked();
		void axeDirClicked();

	private :
		QGroupBox* mainBox;
		QComboBox* imageDirCombo1;
		QComboBox* imageDirCombo2;
		QPushButton* imageDirButton;
		QVector<QLineEdit*> pointsEdit;
		QRadioButton* radioX;	
		QRadioButton* radioY;	
		QRadioButton* radioMX;	
		QRadioButton* radioMY;	

		PaintInterfSegment* paintInterfSegment;
		QString dir;
		const ParamMain* paramMain;
		Assistant* assistant;	
		int nbList;
};

class EchelleWidget : public QWidget
{
	Q_OBJECT

	public :
		EchelleWidget(const ParamMain* pMain, int N, const QStringList & liste1, const QStringList & liste2, Assistant* help, const std::pair<QVector<QString>,QVector<QPoint> > & paramPrec/*=std::pair<QVector<QString>,QVector<QPoint> >()*/, double mesPrec=0);	//pour micmac ortho : liste1=MNT, liste2=ortho
		~EchelleWidget();

		QGroupBox* getBox();
		void updateParam(ParamApero* parametres);
		void updateParam(CarteDeProfondeur* parametres);
		void updateListe2(const QStringList& l);

	signals:
		void updateParam();

	private slots :
		void imageEchClicked();

	private :
		QGroupBox			*mainBox;
		QVector<QComboBox*>  imageEchCombo;
		QPushButton			*imageEchButton;
		QVector<QLineEdit*>  pointsEdit;
		QLineEdit			*distEdit;	

		QVector<PaintInterfSegment*>  paintInterfSegment;
		QString 					  dir;
		const ParamMain				 *paramMain;
		Assistant					 *assistant;	
		int							  nbList;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ImgToOriTabA : public QWidget
{
//onglet pour la saisie de l'image maîtresse
	Q_OBJECT

	public:
		ImgToOriTabA(ParamApero* paramApero, QString dossier, const QVector<ParamImage>* correspImgCalib);//, const QVector<QIcon>* vignettes
		~ImgToOriTabA();

	signals:
		void imgsSetChanged();

	private slots:
		void imageSelected();
		void imageSelected2();
		void addClicked();
		void removeClicked();

	private:
		QString dir;
		ParamApero* parametres;
		//const QVector<QIcon>* icones;

		QListWidget* listWidget;
		QListWidget* listWidget2;
		QPushButton* addButton;	
		QPushButton* removeButton;	
};

class MaitresseTabA : public QWidget
{
//onglet pour la saisie de l'image maîtresse
	Q_OBJECT

	public:
		MaitresseTabA(ParamApero* paramApero, const ParamMain* pMain, QString dossier);
		~MaitresseTabA();

	private slots:
		void maitresseSelected();
		void imgsSetChanged();

	private:
		QString calculeBestMaitresse ();

		QString dir;
		ParamApero* parametres;
		const ParamMain* paramMain;

		QListWidget* listWidget;
		QToolButton* apercuButton;	
};


class ReferenceTabA : public QScrollArea
{
//onglet pour le référencement partiel ou total
	Q_OBJECT

	public:
		ReferenceTabA(ParamApero* paramApero, InterfApero* parentWindow, const ParamMain* pMain, Assistant* help);
		~ReferenceTabA();

		//bool masqueTrouve();
		//void saveMasques();
		bool getDistance();
		QString saveImgAbsParam();
		bool renameDirBDDC();

	signals:
		void imgsSetChanged();

	private slots:
		void radioClicked(int idx);
		void updateParam(int);
		void doPlanDirChecked();
		void doEchelleChecked();
		void radioAbsChecked(bool);
		void fichierAbsClicked();
		void appuisClicked();
		void mesAppClicked();
		void saisieAppClicked();
		void sommetsClicked();
		void filterSelected(const QString& filtre);

	private:
		void resizeEvent(QResizeEvent*);

		QString 		 dir;
		ParamApero      *parametres;
		const ParamMain *paramMain;
		Assistant		*assistant;
		InterfApero		*parent;

		QWidget		 *resizableWidget;
		QRadioButton *radioAucun;
		QRadioButton *radioPlan;
		QRadioButton *radioImageAbs;
		QRadioButton *radioAppuis;
		QRadioButton *radioSommets;
			//orientation manuelle
		QGroupBox 	  *manuBox;
		QCheckBox 	  *checkDoPlanDir;
		QCheckBox 	  *checkDoEchelle;
		QSignalMapper *mapper;
				//masque+direction
		QGroupBox		*planDirBox;
		MasqueWidget	*masqueWidget;
		DirectionWidget *directionWidget;
				//échelle
		EchelleWidget *echelleWidget;
			//orientation absolue d'une image
		QGroupBox 	 		*imgAbsBox;
		QComboBox 	 		*imgAbsCombo;
		QRadioButton 		*radioFichier;
		QRadioButton 		*radioHand;
		QLineEdit	 		*fichierAbsEdit;
		QGroupBox	 		*fichierAbsBox;
		QGroupBox			*formAbsBox;
		QVector<QLineEdit*>  centerAbsEdit;
		QVector<QLineEdit*>  rotationAbsEdit;
		EchelleWidget		*echelleWidget2;
			//géoréférencement par points d'appui
		QGroupBox   	 *appuiBox;
		QLineEdit		 *fileAppEdit;
		QPushButton 	 *fileAppButton;
		QLineEdit		 *fileMesEdit;
		QPushButton 	 *fileMesButton;	
		QPushButton 	 *fileSaisieButton;
		PaintInterfAppui *paintInterfAppui;
			//géoréférencement par coordonnées GPS des sommets
		QGroupBox   *sommetsBox;
		QLineEdit   *fileSommetsEdit;
		QPushButton *fileSommetsButton;		
		FileDialog  *fileDialogSommets;	
};


class OriInitTabA : public QWidget
{
//onglet pour la saisie des orientations initiales
	Q_OBJECT

	public:
		OriInitTabA(ParamApero* paramApero, QString dossier);
		~OriInitTabA();


	private slots:
		void boxChecked();
		void dirClicked();

	private:
		QString dir;
		ParamApero* parametres;

		QCheckBox* checkBox;
		QGroupBox* oriBox;
		QLineEdit* textEdit;
		QPushButton* dirButton;
};


class AutoCalibTabA : public QWidget
{
//onglet pour la saisie de l'image maîtresse
	Q_OBJECT

	public:
		AutoCalibTabA(ParamApero* paramApero, QString dossier, const QVector<ParamImage>* correspImgCalib);//, const QVector<QIcon>* vignettes
		~AutoCalibTabA();

		bool doAutoCalib() const;

	private slots:
		void radioClicked();
		void imageSelected();
		void imageSelected2();
		void addClicked();
		void removeClicked();

	private:
		QString dir;
		ParamApero* parametres;
		const QVector<ParamImage>* images;
		//const QVector<QIcon>* icones;

		QRadioButton* radioOui;
		QRadioButton* radioNon;
		QGroupBox* autoBox;
		QListWidget* listWidget;
		QListWidget* listWidget2;
		QPushButton* addButton;	
		QPushButton* removeButton;	
};

class MultiEchelleTabA : public QWidget
{
//onglet pour le traitement multiéchelle par bloc
	Q_OBJECT

	public:
		MultiEchelleTabA(ParamApero* paramApero, const QList<std::pair<QString, int> >& calibFiles);
		~MultiEchelleTabA();

	public slots:
		void displayMulti();
		void dispList2();

	private:
		const QList<std::pair<QString, int> >* const calibrations;
		ParamApero* parametres;

		QCheckBox* checkMulti;
		QListWidget* listWidget1;
		QListWidget* listWidget2;
		QGroupBox* multiBox;
};

class LiberCalibTabA : public QWidget
{
//onglet pour le libérer les calibrations (une calibration différente par image)
	Q_OBJECT

	public:
		LiberCalibTabA(ParamApero* paramApero, const QList<std::pair<QString, int> >& calibFiles);
		~LiberCalibTabA();

	public slots:
		void liberClicked();

	private:
		ParamApero* parametres;
		const QList<std::pair<QString, int> >* const calibrations;

		QCheckBox** checkLiber;
};

class PtsHomolTabA : public QWidget
{
//onglet pour le libérer les calibrations (une calibration différente par image)
	Q_OBJECT

	public:
		PtsHomolTabA(ParamApero* paramApero);
		~PtsHomolTabA();

	public slots:
		void filtrClicked();
		void calcPt3DClicked();
		void exportPt3DClicked();

	private:
		ParamApero* parametres;

		QCheckBox* checkFiltr;
		QCheckBox* checkCalc3D;
		QCheckBox* checkExport3D;
};


class UserOrientation {
	public :
		UserOrientation();
		UserOrientation(const UserOrientation& userOrientation);
		~UserOrientation();

		UserOrientation& operator=(const UserOrientation& userOrientation);

		const QString getMasque() const;
		const QString getRefMasque() const;

		int getOrientMethode() const;
		bool getBascOnPlan() const;
		const QString& getImgMasque() const;
		const QString& getImage1() const;
		const QString& getImage2() const;
		const QPoint& getPoint1() const;
		const QPoint& getPoint2() const;
		const QPoint& getAxe() const;
		bool getFixEchelle() const;
		const QVector<QPoint>& getPoints() const;
		QVector<QPoint>& modifPoints();
		const QVector<QString>& getImages() const;
		QVector<QString>& modifImages();
		double getDistance() const;
		const QString& getImageGeoref() const;
		const QString& getGeorefFile() const;
		const QVector<REAL>& getCentreAbs() const;
		const QVector<REAL>& getRotationAbs() const;
		const QString& getPointsGPS() const;
		const QString& getAppuisImg() const;

		void setOrientMethode(int om);
		void setBascOnPlan(bool b);
		void setImgMasque(const QString& m);
		void setImage1(const QString& i);
		void setImage2(const QString& i);
		void setPoint1(const QPoint& p);
		void setPoint2(const QPoint& p);
		void setAxe(const QPoint& p);
		void setFixEchelle(bool f);
		void setPoints(const QVector<QPoint>& l);
		void setImages(const QVector<QString>& l);
		void setDistance(double d);
		void setImageGeoref(const QString& i);
		void setGeorefFile(const QString& g);
		void setCentreAbs(const QVector<REAL>& c);
		void setRotationAbs(const QVector<REAL>& r);
		QVector<REAL>& modifCentreAbs();
		QVector<REAL>& modifRotationAbs();
		void setPointsGPS(const QString& file);
		void setAppuisImg(const QString& file);

	private :
		void copie(const UserOrientation& userOrientation);

		int orientMethode;	//0 : en l'air, 1 : plan+direction+échelle, 2 : orientation absolue d'une image , 3 : points d'appui, 4 : sommets GPS
		//orientation manuelle
			//plan
		bool bascOnPlan;
		QString imgMasque;	//first : image, second : masque=image+section(".",0,-2)+QString("_MasqPlan.tif"), réf=second.section(".",0,-2)+QString(".xml")
			//direction
		QString image1;
		QString image2;
		QPoint point1;
		QPoint point2;
		QPoint axe;
			//echelle
		bool fixEchelle;	//il doit au moins y avoir soit plan+direction soit l'échelle de fixé
		QVector<QPoint> points;	//point1 img1, point2 img1, point1 img2, point2 img2
		QVector<QString> images;
		double distance;
		//orientation absolue d'une image
		QString imageGeoref;	//image géoréférencée
		QString georefFile;	//si par un fichier (chemin complet)
		QVector<REAL> centreAbs;	//si saisis
		QVector<REAL> rotationAbs;
		//géoréférencement
		QString pointsGPS;	//coordonnées GPS des points d'appui (méthode 3) ou fichier/dossier des sommets GPS (méthode 4)
		QString appuisImg;
};
class ParamApero
{
	public :
		ParamApero();
		ParamApero(const ParamApero& paramApero);
		~ParamApero();

		ParamApero& operator=(const ParamApero& paramApero);

		const QStringList& getImgToOri() const;
		QStringList& modifImgToOri();
		const QString& getImgMaitresse() const;
		const UserOrientation& getUserOrientation() const;
		UserOrientation& modifUserOrientation();
		bool getUseOriInit() const;
		const QString& getDirOriInit() const;
		const QStringList& getAutoCalib() const;
		QStringList& modifAutoCalib();
		bool getMultiechelle() const;
		const QList<int>& getCalibFigees() const;
		QList<int>& modifCalibFigees();
		const QVector<bool>& getLiberCalib() const;
		QVector<bool>& modifLiberCalib();
		bool getFiltrage() const;
		bool getCalcPts3D() const;
		bool getExportPts3D() const;

		void setImgToOri(const QStringList& l);
		void setImgMaitresse(const QString& s);
		void setUserOrientation(const UserOrientation& uo);
		void setUseOriInit(bool u);
		void setDirOriInit(const QString& d);
		void setAutoCalib(const QStringList& l);
		void setMultiechelle(bool b);
		void setCalibFigees(const QList<int>& cf);
		void setLiberCalib(const QVector<bool>& lc);
		void setFiltrage(bool b);
		void setCalcPts3D(bool b);
		void setExportPts3D(bool b);

	private :
		void copie(const ParamApero& paramApero);

		QStringList		imgToOri;
		QString			imgMaitresse;
		UserOrientation userOrientation;
		bool		 	useOriInit;
		QString		 	dirOriInit;
		QStringList	 	autoCalib;
		bool			multiechelle;
		QList<int>		calibFigees;
		QVector<bool> 	liberCalib;
		bool			filtrage;
		bool			calcPts3D;
		bool			exportPts3D;
};

class InterfApero : public QDialog
{
	Q_OBJECT

	public:
		InterfApero(const ParamMain* pMain, QWidget* parent, Assistant* help);
		~InterfApero();

		//récupération des données
		const ParamApero& getParamApero() const;
		bool isDone();

	public slots:
		void calcClicked();
		void precClicked();
		void suivClicked();
		void tabChanged();
		void helpClicked();

	private:
		void updateInterfApero(int tab);

		//QVector<QIcon> icones;
		QTabWidget		 *tabWidget;
		ImgToOriTabA	 *imgToOriTabA;
		MaitresseTabA	 *maitresseTabA;
		ReferenceTabA	 *referenceTabA;
		OriInitTabA		 *oriInitTabA;
		AutoCalibTabA	 *autoCalibTabA;
		MultiEchelleTabA *multiEchelleTabA;
		LiberCalibTabA	 *liberCalibTabA;
		PtsHomolTabA	 *ptsHomolTabA;
		QPushButton 	 *precButton;
		QPushButton 	 *calButton;
		QPushButton 	 *cancelButton;
		QPushButton 	 *helpButton;
		Assistant		 *assistant;
		bool 			  done;

		//données
		const ParamMain  *paramMain;
		ParamApero 		  paramApero;
};

#endif
