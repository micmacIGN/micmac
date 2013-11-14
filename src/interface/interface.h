/*
interface.h et interface.cpp correspondent à la fenêtre principale de l'interface.
La classe ParamMain regroupe les paramètres généraux utiles à travers l'ensemble du traitement du chantiers : avancement du calcul, données utilisateurs, noms des fichiers entrées/sorties des calculs. ParamMain regroupe aussi certaines fonctions récurrentes de recherche dans les données. Ce sont les données de ParamMain qui sont enregistrées à chaque fermeture de l'interface ; elles permettent la reprise du calcul au même point ou de recommencer une étape.
La classe ParamImage regroupe l'ensemble des données sur chaque image.
La classe Interface correspond à la fenêtre principale. Elle regroupe l'initialisation des fenêtres secondaires et des paramètres à chaque changement de chantier, les affichages sur l'avancement du calcul et les menus, chacun associé à une fonction donnant accès à une fenêtre secondaire puis déclenchant le calcul associé via les threads (voir applithread.h). Lors d'un calcul, la barre de progression Interface::progress indique la progression du calcul dans le thread principal et est mise à jour chaque seconde via un QTimer dans la fonction updateProgressDialog. Lorsque le thread est fini (calcul fini ou échoué), la mise à jour est gérée par la fonction threadFinished.
Quelques fonctions globales récurrentes gèrent les chaîntes de caractères, les info système ou les dossiers.
*/


#ifndef INTERFACE_H
#define INTERFACE_H

#include "all.h"

#if defined Q_WS_MAC
    #include <sys/sysctl.h>
#endif


bool rm(const QString& directory);
void deleteFile(const QString& file, bool dir=false);
QString dirBin(const QString& micmacdir);
bool checkPath(const QString& path);
int execute(QString commande);
QString applicationPath();	//sans le "/"
QString systemeNumerique(QString& virgule);
QString systemeNumerique(QString& virgule, QString& point);

const char* ch(const QString& s);
ostream& ecr(const QString& s);
QString conv(const QString& s);
QString conv(const char* c);
void qMessageBox(QWidget* w, QString title, QString msg);
QString remplace(const QString& s, const QString& before, const QString& after);
QString comm(const QString& s);
int killall(const char* programme);
void print(const QString& s);
QString noBlank(const QString& s);

extern QString g_interfaceDirectory;
extern QString g_iconDirectory;

class FileDialog : public QFileDialog
{
//boîte de dialog traduite
	Q_OBJECT

	public:
		FileDialog(QWidget* parent=0, const QString& caption=QString(), const QString& directory=QString(), const QString& filter=QString());
		~FileDialog();
};

class ParamImage {
	public :
		ParamImage();
		ParamImage(const ParamImage& paramImage);

		ParamImage& operator=(const ParamImage& paramImage);
		bool isEqualTo(const QString& image, int type) const;
		bool isEqualTo(const ParamImage& paramImag, int type) const;

		const QString& getImageRAW() const;
		const QString& getImageTif() const;
		const QString& getCalibration() const;
		const QSize& getTaille() const;
		QString getNumero() const;

		void setImageRAW(const QString& raw);
		void setImageTif(const QString& tif);
		void setCalibration(const QString& calib);
		void setTaille(const QSize& size);
		bool calcNumero(const ParamMain& paramMain);

		static std::pair<int,int> numPos;	//pos 1ier char changeant à partir du début et pos 1ier char changeant compté à partir de la fin

	private :
		void copie(const ParamImage& paramImage);

		QString imageRAW;
		QString imageTif;
		QString calibration;
		QSize taille;
		QString numero;
};

class ParamMain
//paramètres principaux du calcul
{
	public :
		enum Mode { BeginMode, ImageMode, PointsEnCours, PointsMode, PoseEnCours, PoseMode, CarteEnCours, EndMode };

		ParamMain();
		ParamMain(const ParamMain& paramMain);
		~ParamMain();

		ParamMain& operator=(const ParamMain& paramMain);

		void init();
		const QStringList& getFormatsImg() const;
		const QVector<std::pair<Mode,QString> >& getTradMode() const;
		const QVector<std::pair<Mode,QString> >& getTradModeInternational() const;

		static bool isValideFormat(const QString& extension);
		QString convertTifName2Couleur(const QString& image) const;
		bool calcImgsId();
		QString getNumImage(const QString& image, bool* ok=0, bool TA=false) const;
		int findImg(const QString& image, int type, bool strict=true) const;	//0 : imageRaw, 1 : imageTif
		QString saveImgsSize();	//récupération des tailles des images et enregistrement dans getCorrespImgCalib
		int nbCartesACalculer() const ;

		bool isFrench() const;
		const QString& getMicmacDir() const;
		const QString& getCalculXML() const;
		const Mode& getCurrentMode() const;
		int getAvancement() const;
		int getEtape() const;
		const QString& getDossier() const;
		const ParamPastis& getParamPastis() const;
		ParamPastis& modifParamPastis();
		static const QString& getImageXML();
		static const QString& getCoupleXML();
		static const QString& getAssocCalibXML();
		const QVector<ParamImage>& getCorrespImgCalib() const;
		QVector<ParamImage>& modifCorrespImgCalib();
		static const QString& getChantierXML();
		const QString& getMakeFile() const;
		const ParamApero& getParamApero() const;
		ParamApero& modifParamApero();
		static const QString& getAperoXML();
		static const QString& getMaitresseXML();
		static const QString& getExportPlyXML();
		static const QString& getCleCalibFishEyeXML();
		static const QString& getCleCalibClassiqXML();;
		static const QString& getContraintesXML();
		static const QString& getImgOriAutoCalibXML();
		static const QString& getImgOriXML();
		static const QString& getCalibDefXML();
		static const QString& getCleCalibCourtXML();
		static const QString& getCleCalibCourtFishEyeXML();
		static const QString& getCleCalibCourtClassiqXML();
		static const QString& getCleCalibLongFishEyeXML();
		static const QString& getCleCalibLongClassiqXML();
		static const QString& getDefCalibVInitXML();
		static const QString& getDefCalibCourtXML();
		static const QString& getImgsOriVInitXML();
		static const QString& getImgsCourtOriXML();
		static const QString& getPosesFigeesXML();
		static const QString& getPosesLibresXML();
		static const QString& getDefCalibTtInitXML();
		static const QString& getImgsOriTtInitXML();
		static const QString& getPosesNonDissocXML();
		static const QString& getCleCalibLiberFishEyeXML();
		static const QString& getCleCalibLiberClassiqXML();
		static const QString& getOrientationGPSXML();
		static const QString& getOrientationAbsolueXML();
		static const QString& getDefObsGPSXML();
		static const QString& getDefIncGPSXML();
		static const QString& getPonderationGPSXML();
		static const QString& getOriInitXML();
		static const QString& getMicmacXML();
		static const QString& getMicmacTerXML();
		static const QString& getMicmacMMOrthoXML();
		const QVector<CarteDeProfondeur>& getParamMicmac() const;
		QVector<CarteDeProfondeur>& modifParamMicmac();
		static const QString& getIntervalleXML();
		static const QString& getDiscontinuteXML();
		static const QString& getDefMasqueXML();
		static const QString& getCartesXML();
		static const QString& getRepereXML();
		static const QString& getNomCartesXML();
		static const QString& getNomTAXML();
		static const QString& getOrthoXML();
		static const QString& getParamPortoXML();
		static const QString& getPortoXML();
		static const QString& getPathMntXML();

		void setFrench(bool f);
		void setMicmacDir(const QString& micmacDossier);
		void setCalculXML(const QString& calculFile);
		void setCurrentMode(Mode mode);
		void setAvancement(int n);
		void setEtape(int n);
		void setDossier(const QString& dir);
		void setCorrespImgCalib(const QVector<ParamImage>& array);
		void setParamPastis(const ParamPastis& pPastis);
		void setMakeFile(const QString& file);
		void setParamApero(const ParamApero& pApero);
		void setParamMicmac(const QVector<CarteDeProfondeur>& p);

	private :
		friend void copie(ParamMain* paramMain1, const ParamMain& paramMain2);

		bool french;
		static QStringList formatsImg;
		QVector<std::pair<Mode,QString> > tradMode;	//en Français seulement ; le fichier de sauvegarde ne peut pas dépendre de la langue (elle peut être changée lors de la reprise du calcul)
		QVector<std::pair<Mode,QString> > tradModeInternational;	//traduction de tradModeInternational en fonction de la langue, permet la représentation dans la MainWindow
		QString micmacDir;

		QString calculXML;	//avec dossier
		Mode currentMode;	//avancement du calcul
		int avancement;
		int etape;	//pour les calculs récursifs (points 3D de AperoThread, cartes de MicmacThread)

		//ImageMode
		QString dossier;
		QVector<ParamImage> correspImgCalib;	//sans le dossier

		//PointsMode
		ParamPastis* paramPastis;
		static QString postfixTifCouleur;
		static QString chantierXML;	//param du chantier
		static QString imageXML;	//sans le dossier
		static QString coupleXML;	//sans le dossier
		static QString assocCalibXML;	//sans le dossier
		QString makeFile;	//makefile utilisé pour pastisThread si non terminé (soit pour MapCmd, soit pour Pastis)

		//PoseMode
		ParamApero* paramApero;
		static QString aperoXML;	//param pour apero
		static QString maitresseXML;	//fichier xml de définition de l'image maîtresse
		static QString exportPlyXML;	//export des points homologues 3D et des caméras au format ply
			//fichiers supplémentaires pour distinguer fish-eye et calib classique
			static QString cleCalibFishEyeXML;
			static QString cleCalibClassiqXML;
			static QString contraintesXML;
			//auto-calibration
			static QString imgOriAutoCalibXML;
			//cas mono-échelle
			static QString imgOriXML;	//fichier xml de la liste des images à orienter
			static QString calibDefXML;
			//fichiers supplémentaires pour le cas multi-échelle
			static QString cleCalibCourtXML;
			static QString cleCalibCourtFishEyeXML;
			static QString cleCalibCourtClassiqXML;
			static QString cleCalibLongFishEyeXML;
			static QString cleCalibLongClassiqXML;
			static QString defCalibVInitXML;
			static QString defCalibCourtXML;
			static QString imgsOriVInitXML;
			static QString imgsCourtOriXML;
			static QString posesFigeesXML;
			static QString posesLibresXML;
			//fichiers supplémentaires pour le cas des calibrations dissociées
			static QString defCalibTtInitXML;
			static QString imgsOriTtInitXML;
			static QString posesNonDissocXML;
			static QString cleCalibLiberFishEyeXML;
			static QString cleCalibLiberClassiqXML;
			//fichiers supplémentaires pour le cas d'une orientation absolue (plan + direction ou image référence ou points d'appui)
			static QString orientationAbsolueXML;
			//fichiers supplémentaires pour le cas d'un géoréférencement avec points GPS
			static QString defObsGPSXML;
			static QString defIncGPSXML;
			static QString ponderationGPSXML;
			//fichiers supplémentaires pour le cas d'un géoréférencement avec sommets GPS
			static QString orientationGPSXML;
			//orientation initiale
			static QString oriInitXML;
			
		//MasqueMode		
		QVector<CarteDeProfondeur> paramMicmac;
				
		//EndMode
		static QString micmacXML;		//param pour micmac (géom image)
		static QString micmacTerXML;	//param pour micmac (géom terrain)
		static QString micmacMMOrthoXML;//param pour micmac (orthoimages seules)
		static QString intervalleXML;
		static QString discontinuteXML;
		static QString defMasqueXML;
		static QString cartesXML;		//liste des images utilisées pour la corrélation pour la carte courante
		static QString repereXML;		//repère du MNT
		static QString nomCartesXML;	//nom des cartes en sortie
		static QString nomTAXML;		//nom du TA en sortie
		static QString orthoXML;		//nom des orthoimages simples en sortie

		static QString paramPortoXML;	//paramètres du mosaïcage (Porto)
		static QString portoXML;		//nom des orthoimages simples en entrée du mosaïcage
		static QString pathMntXML;		//mnt de la carte de profondeur pour l'orthoimage mosaïquée
};


class Timer : public QTime
{
	public :
		Timer(QString dossier);
		void displayTemps(QString label);	

	private :
		QString dir;
};


class Interface : public QMainWindow
//fenêtre principale : importation des images, ouverture d'un calcul en cours, accès aux autres fenêtres, transfert de leurs paramètres pour écriture et enregistrement du calcul en cours
{
   Q_OBJECT

	public:
		Interface(QSettings& globParam);
		~Interface();

		static int getCpuCount();
		void displayErreur();
		void setStdOutFile(const QString& fichier);
		ParamMain& modifParamMain();
		static int dispMsgBox(const QString& info, const QString& question, QVector<int> reponses, int defaut=2);	//réponses : boutons oui, non et annuler ; 0=accept (oui), 1=reject (non), 2=destruction (annuler), -1=pas de bouton

	public slots:
		void openCalc(const QString& fichier=QString());
		void saveCalcAs();

	private slots:
		void openImg();
		void saveCalc(QString file=QString(), bool msg=true);
		void saveCalcSsMsg();
		bool closeAppli();
		void calcPastis(bool continuer=false);
		void calcApero(bool continuer=false);
		void calcMicmac(bool continuer=false);
		void continueCalc();
		void vueHomol();
		void vue();
		void vueNuages();
		void cartesProf8B();
		void modeles3D();
		void orthoimage();
		void help();
		void about();
		void options();
		void verifMicmac();
		void nettoyage();
		void supprImg();
		void progressCanceled();
		void threadFinished();
		void updateProgressDialog ();

	private:
		enum Msg { Save, ImageDir, ErasePt, ErasePose, EraseCarte};

		void createActions();
		void createMenus();
		void updateInterface(ParamMain::Mode mode);
		void contextMenuEvent(QContextMenuEvent *event);
		QSize sizeHint ();
		QSize minimumSizeHint ();
		QSize maximumSizeHint ();
		void resizeEvent(QResizeEvent* event);
		void initialisation();
		bool checkMicmac();
		QString pathToMicMac (QString currentPath, const QStringList& l, QString lastDir);
		void setUpCalc();
		void closeEvent(QCloseEvent* event);

		QMenuBar* menuBarre;
		QLabel* topLabel;
		QToolButton* topLogos;
		QMenu *fileMenu;
		QMenu *calcMenu;
		QMenu *visuMenu;
		QMenu *convertMenu;
		QMenu *helpMenu;
		QAction *openCalcAct;
		QAction *openImgAct;
		QAction *saveCalcAct;
		QAction *saveCalcAsAct;
		QAction *exitAct;
		QAction *calcPastisAct;
		QAction *calcAperoAct;
		QAction *calcMicmacAct;
		QAction *continueCalcAct;
		QAction *supprImgAct;
		QAction *vueHomolAct;
		QAction *vueAct;
		QAction *vueNuageAct;
		QAction *prof8BAct;
		QAction *mod3DAct;
		QAction *orthoAct;
		QAction *helpAct;
		QAction *aboutAct;
		QAction *optionAct;
		QAction *verifMicmacAct;
		QAction *nettoyageAct;
		Assistant* assistant;
    		QLabel *chantierLabel;	
    		QLabel *pastisLabel;	
    		QLabel *aperoLabel;	
    		QLabel *micmacLabel;	
    		QLabel *infoLabel;
    		QTreeWidget *imagesList;	//affichage de la liste des images
		QProgressDialog* progress;
		Progression* progressBar;
		QTimer* timer;

		QSettings* settings;
		QString defaultDir;
		ParamMain paramMain;
		bool saved;	//si le calcul est modifié et non enregistré
		static int cpu_count;	//nombre de processeurs de l'ordinateur (cpu_count >= maxcpu)
		int maxcpu;	//nombre de processeurs à utiliser
		InterfPastis* interfPastis;	//interface pour saisir les paramètres de Pastis
		InterfApero* interfApero;	//interface pour saisir les paramètres de Apero
		InterfMicmac* interfMicmac;	//interface pour saisir les paramètres de Mcimac
		VueHomologues* vueHomologues;	//interface pour visualiser les points homologues
		VueChantier* vueChantier;	//interface pour visualiser les paramètres du chantier
		VueChantier* vueCartes;		//interface pour visualiser les nuages de points
		InterfCartes8B* interfCartes8B;		//interface pour saisir les paramètres de GrShade
		InterfModele3D* interfModele3D;		//interface pour saisir les paramètres de Nuage2Ply
		InterfOrtho* interfOrtho;		//interface pour saisir les paramètres de l'orthoimage
		InterfOptions* interfOptions;		//interface pour changer les paramètres globaux de l'interface
		InterfVerifMicmac* interfVerifMicmac;	//interface pour vérifier le bon déroulement du calcul Micmac
		AppliThread* appliThread;	//calculs par thread
		bool annulation;	//true si la QProgressDialog a été annulée
		QString stdoutfilename;
		//fpos_t* posReadError;
		long int posReadError;
};

#endif
