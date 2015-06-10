/*
interfPastis.h et interfPastis.cpp correspondent à la fenêtre Calcul -> Points homologues qui permet de convertir les images aux format tif, de leur associer la calibration interne initiale correspondante et de calculer les points homologues de couples d'images.
La classe InterfPastis correspond à la fenêtre générale.
Cette fenêtre est constituée de plusieurs onglets ; elle permet :
- de définir le format des noms des images et de créer/importer des fichiers de calibration initiales -> onglet CalibTab ;
	* La GroupBox imgNameBox permet de saisir la décomposition du nom des images ; la classe ImageName regroupe les paramètres du nom des images (préfixe, postfixe, extension et taille du pixel), le fichiers xml/BDCamera.xml est une base de données qui recense les caméras utilisées (objets ImageName) ;
	* La GroupBox calibBox permet de saisir les calibrations internes initiales pour chaque objectif utilisé ; la classe CalibCam regroupe les paramètres d'une calibration (focale, PP, distorsion...).
	Pour ces deux paramètres, il est possible de sélectionner une valeur/ouvrir un fchier existant(e), supprimer une valeur ou en créer une à l'aide d'un formulaire (formImgBox et formCalibBox).
- de sélectionner le type de chantier (convergent, par bandes, divergent...) -> onglet ChoiceTab;
- de sélectionner les couples d'images dans lesquels chercher des points homologues -> onglet CoupleTab (et ses sous-classes, cet onglet dépendra du type de chantier mais seul CoupleTabConv est programmé); les couples encore possibles sont listés dans 2 QListWidget (image1 et image2 possible) et les couples sont listés et classés dans un QTreeWidget, ces 3 listes sont mises à jour à chaque ajout/supppression d'un ou plusieurs couples ;
- de saisir les autres paramètres pour Pastis (taille de l'image sous-échantillonnée) -> onglet CommunTabP.
La classe ParamPastis enregistre tous les paramètres saisis.
*/

#ifndef INTERFPASTIS_H
#define INTERFPASTIS_H

#include "all.h"

class InterfPastis;
class ParamPastis;

class CalibCam
{
	public :
		CalibCam ();
		CalibCam (int modele, const QString& fileName, double f, double px, const QPointF& PPApt=QPointF(-1,-1), const QSize& size=QSize(0,0), const QPointF& PPSpt=QPointF(-1,-1), const QVector<double>& dist=QVector<double>(4,0), int rUtil=0, const QVector<double>& pRad=QVector<double>());
		CalibCam (const QString& fileName);
		CalibCam (const CalibCam& calibCam);
		~CalibCam ();

		const QString& getFile() const;
		int getType() const;
		double getFocale() const;
		double getFocalePx() const;
		double getTaillePx() const;
		const QPointF& getPPA() const;
		const QSize& getSizeImg() const;
		const QPointF& getPPS() const;
		const QVector<double>& getDistorsion() const;
		int getRayonUtile() const;
		const QVector<double>& getParamRadial() const;

		void setFocale (int f);

		CalibCam& operator=(const CalibCam& calibCam);
		bool operator==(const CalibCam& calibCam) const;
		bool operator==(const QString& fileName) const;

		bool setDefaultParam (const QSize& tailleImg);

	private :
		void copie(const CalibCam& calibCam);

		QString 		file;	//sans le dossier
		int 			type; //0 : classique, 1 : fish-eye
		double 			focale;	//en mm
		double 			taillePx;	//en microns
		QPointF 		PPA;
		QSize 			sizeImg;
		QPointF 		PPS;
		QVector<double> distorsion;	
		int 			rayonUtile;
		QVector<double> paramRadial;	
};
class CalibTab : public QWidget
{
	Q_OBJECT

	public:
		CalibTab(InterfPastis* interfPastis, ParamPastis* parametres, ParamMain* pMain, const QList<int>& focales, const QList<QSize>& formatCalibs, const QList<int>& refImgCalibs);//bool tifOnly, 
		~CalibTab();

		void resizeTab();
		bool allCalibProvided();
		void updateTab(bool show, bool resize=true);

		const QList<int>& getCalibAFournir() const;
		const QList<QSize>& getFormatCalibAFournir() const;

	private slots :
		void addCalibClicked();
		void removeCalibClicked();
		void newCalibClicked();
		void radioClicked();
		void saveNewCalibClicked();
		void cancelNewCalibClicked();
		void focaleEdited();
		void taillePxEdited();
		void tailleCapteurEdited();
		void camChanged();

	private:
		double qstringToDouble(QString coeff, bool* ok=0);
		void removeFromList(const QStringList& items);

		QListWidget	 *calibViewSelected;
		QPushButton	 *addCalib;
		QPushButton  *removeCalib;
		QPushButton  *newCalib;
		QRadioButton *radioClassique;
		QRadioButton *radioFishEye;
		QLineEdit	 *focaleEdit;
		QLineEdit	 *taillePxEdit;
		QLineEdit	 *tailleCapteurEdit;
		QComboBox	 *camCombo;
		QLineEdit	 *PPAXEdit;
		QLineEdit	 *PPAYEdit;
		QLineEdit	 *sizeWEdit;
		QLineEdit	 *sizeHEdit;
		QLineEdit	 *PPSXEdit;
		QLineEdit	 *PPSYEdit;
		QLineEdit	 *distorsionaEdit;
		QLineEdit	 *distorsionbEdit;
		QLineEdit	 *distorsioncEdit;
		QLineEdit	 *distorsiondEdit;
		QGroupBox	 *classiqueBox;
		QLineEdit	 *rayonEdit;
		QTextEdit	 *paramRadialEdit;
		QGroupBox	 *fishEyeBox;
		QPushButton	 *saveNewCalib;
		QPushButton	 *cancelNewCalib;
		QGroupBox	 *formCalibBox;
		QGroupBox	 *calibBox;

		InterfPastis 	   *parent;
		ParamPastis	 	   *paramPastis;
		QString 	 	    dir;
		ParamMain		   *paramMain;
		const QList<int>	calibAFournir;
		const QList<QSize>	formatCalibAFournir;
		const QList<int> 	refImgCalibAFournir;
		//bool noCam;
		QList<std::pair<QString,double> > imgNames;	//taille de pixel des caméras (BDCamera) -> ne pas modifier (modification dans Assistant.cpp)
		static CalibCam 				  defaultCalib;
		static QString 					  defaultCalibName;	//préfix par défaut des fichier.xml de calibrations internes créés
		int 							  longueur;
};

class ChoiceTab : public QWidget
{
	Q_OBJECT

	public:
		ChoiceTab(InterfPastis* interfPastis, ParamPastis* parametres);
		~ChoiceTab();

	signals:
		void typechanged();

	private slots:
		void radioClicked();

	private:
		QRadioButton *radioConvergent;
		QRadioButton *radioBandes;
		QRadioButton *radioAutre;
		InterfPastis* parent;
		ParamPastis* paramPastis;
};

class CoupleTab : public QWidget
{
	Q_OBJECT

	public:
		CoupleTab(InterfPastis* interfPastis, ParamMain* paramMain, ParamPastis* parametres);
		~CoupleTab();

		bool isDone();
		void getAllCouples() const;	//retourne tous les couples possibles (cas parallèle), indépendamment de ce qui a été saisi

	private slots :
		void addTheseClicked();
		void addAllButtonClicked();
		void addAllClicked();
		void addKNearestClicked();
		void removeClicked();
		void removeAllClicked();
		void expandAll();
		void collapseAll();
		void list1Selected();
		void list2Selected();
		void treeWidgetSelected();

	private:
		void initList(QListWidget* listWidget);
		void updateCoupleTab ();

		QListWidget *listWidget1;
		QListWidget *listWidget2;
		QTreeWidget *treeWidget;
		QToolButton *addButton;
		QAction *addAllAct;
		QAction *addKNearestAct;
		QPushButton *addAllButton;
		QPushButton *removeButton;
		QPushButton *removeAllButton;
		QPushButton *expandCollapseButton;
		QGroupBox* groupBox;

		InterfPastis* parent;
		ParamPastis* paramPastis;
		QString dir;
		const QVector<ParamImage>* correspImgCalib;
		bool done;
};

class CommunTabP : public QWidget
{
	Q_OBJECT

	public:
		CommunTabP(InterfPastis* interfPastis, ParamPastis* parametres, int largeurMax);
		~CommunTabP();
	
		bool getLargeurMaxText(int passe) const;	//1 ou 2

	public slots:
		void updateMultiscale();

	private slots:
		void seuilChanged();

	private:
		QCheckBox *checkMultiscale;
		QLabel	  *largeurMax1Label;
		QLabel	  *largeurMax2Label;
		QLineEdit *largeurMax1Edit;
		QLineEdit *largeurMax2Edit;
		QLabel	  *seuilLabel;
		QSpinBox  *seuilBox;

		int			  tailleMax;	//taille réelle des images
		InterfPastis *parent;
		ParamPastis	 *paramPastis;		
};

class ParamPastis
{
	public :
		enum TypeChantier { Convergent=1, Bandes};

		ParamPastis();
		ParamPastis(const ParamPastis& paramPastis);
		~ParamPastis();

		static const QVector<std::pair<TypeChantier,QString> >& getTradTypChan();
		static const QVector<std::pair<TypeChantier,QString> >& getTradTypChanInternational();
		int findFocale(const QString& calibFile) const;
		static bool extractFocaleFromName(const QString& calibFile, int& focale);

		const TypeChantier& getTypeChantier() const;
		int getLargeurMax() const;
		const QList<CalibCam>& getCalibs() const;
		const QList<std::pair<QString, int> >& getCalibFiles() const;
		const QList<std::pair<QString, QString> >& getCouples() const;
		bool getMultiScale() const;
		int getLargeurMax2() const;
		int getNbPtMin() const;

		void setTypeChantier(const TypeChantier& typChan);
		void setLargeurMax(int l);
		QList<CalibCam>& modifCalibs();
		QList<std::pair<QString, int> >& modifCalibFiles();
		QList<std::pair<QString, QString> >& modifCouples();
		void setMultiScale(bool b);
		void setLargeurMax2(int l);
		void setNbPtMin(int n);

		ParamPastis& operator=(const ParamPastis& paramPastis);

	private :
		void copie(const ParamPastis& paramPastis);
		static void fillTradTypChan();

		static QVector<std::pair<TypeChantier,QString> > tradTypChan;
		static QVector<std::pair<TypeChantier,QString> > tradTypChanInternational;

		TypeChantier 						typeChantier;
		QList<CalibCam> 					calibs;		//sans le dossier : toutes les calibrations (pour Apero)
		QList<std::pair<QString, int> > 	calibFiles;	//sans le dossier : toutes les calibrations
		int									largeurMax;
		QList<std::pair<QString, QString> > couples;
		bool 								multiscale;	//true si recherche multi-échelle
		int 								largeurMax2;
		int 								nbPtMin;
};


class InterfPastis : public QDialog
{
	Q_OBJECT

	public:
		InterfPastis(QWidget* parent, Assistant* help, ParamMain* pmMain);
		~InterfPastis();

		//récupération des données
		const ParamPastis& getParamPastis() const;
		void relaxSize();
		void adjustSizeToContent();
		bool isDone();

	private slots:
		void calcClicked();
		void precClicked();
		void suivClicked();
		void helpClicked();
		void tabChanged();
		void typeChantierChanged();

	private:
		void updateInterfPastis(int tab);

		QTabWidget	*tabWidget;
		CalibTab	*calibTab;
		ChoiceTab	*choiceTab;
		CoupleTab	*coupleTab;
		CommunTabP	*communTab;
		QPushButton *precButton;
		QPushButton *calButton;
		QPushButton *cancelButton;
		QPushButton *helpButton;
		Assistant	*assistant;
		bool 		 done;

		//données
		static int				   numChoiceTab;
		ParamMain				  *paramMain;
		ParamPastis 			   paramPastis;
		QString 				   dossier;
		ParamPastis::TypeChantier  oldTypeChantier;	
		ParamMain				  *precParamMain;
		const QVector<ParamImage> *correspImgCalib;
		int 					   longueur;
};

#endif
