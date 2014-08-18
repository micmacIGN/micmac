#if defined Q_WS_WIN 
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "appliThread.h"

#define GLWidget GLWidget_IC

using namespace std;


AppliThread::AppliThread () :
	QThread(),
	paramMain( 0 ),
	endResult( 0 ),
	annulation( 0 ),
	progressLabel( QString() ),
	readerror(),
	micmacDir( QString() ),
	stdoutfile( QString() ),
	done( true ){
}

AppliThread::AppliThread (ParamMain* pMain, const QString& stdoutfilename, bool* annul) :
	QThread(),
	paramMain( pMain ),
	annulation( annul ),
	progressLabel( QString() ),
	readerror(),
	micmacDir( QString() ),
	stdoutfile( stdoutfilename ),
	done( true ){
	if (pMain!=0) micmacDir = pMain->getMicmacDir();
}

AppliThread::AppliThread(const AppliThread& appliThread) : QThread() { copie(this,appliThread); }
AppliThread::~AppliThread () {}

AppliThread& AppliThread::operator=(const AppliThread& appliThread) {
	if (&appliThread!=this)
		copie(this,appliThread);
	return *this;
}

void copie(AppliThread* appliThread1, const AppliThread& appliThread2) {
	appliThread1->readerror = appliThread2.readerror;
	appliThread1->micmacDir = appliThread2.micmacDir;
	appliThread1->paramMain = appliThread2.paramMain;
	appliThread1->progressLabel = appliThread2.progressLabel;
	appliThread1->annulation = appliThread2.annulation;
	appliThread1->stdoutfile = appliThread2.stdoutfile;
	appliThread1->done = appliThread2.done;
}
		
void AppliThread::run() {}	
bool AppliThread::killProcess() { return false; }	

const QString& AppliThread::getReaderror() const { return readerror;}
int AppliThread::getEndResult() const { return endResult;}
const QString& AppliThread::getProgressLabel() const { return progressLabel;}
const QString& AppliThread::getStdoutfile() const { return stdoutfile;}
ParamMain* AppliThread::getParamMain() { return paramMain;}
const QString& AppliThread::getMicmacDir() const { return micmacDir;}
bool* AppliThread::getAnnulation() { return annulation;}
bool AppliThread::getIsDone() const { return done; }

void AppliThread::setEndResult(int endRes) { endResult=endRes;}
void AppliThread::setProgressLabel(const QString& label) { progressLabel=label;}
void AppliThread::setReaderror(const QString& error) { readerror=error;}
void AppliThread::setDone(bool b) { done=b;}
				
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


PastisThread::PastisThread(ParamMain* pMain, const QString& stdoutfilename, bool* annul, int cpu) : AppliThread(pMain,stdoutfilename,annul), cpu_count(cpu)
	{ setEndResult(Termine); }
PastisThread::PastisThread(const PastisThread& pastisThread) : AppliThread() { copie(this,pastisThread); }
PastisThread::~PastisThread () {}

PastisThread& PastisThread::operator=(const PastisThread& pastisThread) {
	if (this!=&pastisThread) copie(this,pastisThread);
	return *this;
}

void copie(PastisThread* pastisThread1, const PastisThread& pastisThread2) {
	copie(dynamic_cast<AppliThread*>(pastisThread1), pastisThread2);
	pastisThread1->cpu_count = pastisThread2.cpu_count;
}

void PastisThread::run() {
	if (getParamMain()->getAvancement()<=Enregistrement) {	
		setProgressLabel(conv(tr("User data saving")));

		//effacement des résultats précédents
		if (QDir(getParamMain()->getDossier()+QString("Pastis")).exists()) {
			bool b = rm(getParamMain()->getDossier()+QString("Pastis"));
			if (!b) {
				setReaderror(QString("Pastis"));
				getParamMain()->setAvancement(-31);
				return;
			}
		}
		if (QDir(getParamMain()->getDossier()+QString("Homol")).exists()) {
			bool b = rm(getParamMain()->getDossier()+QString("Homol"));
			if (!b) {
				setReaderror(QString("Homol"));
				if (!*getAnnulation()) getParamMain()->setAvancement(-31);
				return;
			}
		}
		if (QDir(getParamMain()->getDossier()+QString("Pastis_Init")).exists()) {
			bool b = rm(getParamMain()->getDossier()+QString("Pastis_Init"));
			if (!b) {
				setReaderror(QString("Pastis_Init"));
				getParamMain()->setAvancement(-31);
				return;
			}
		}
		if (QDir(getParamMain()->getDossier()+QString("Homol_Init")).exists()) {
			bool b = rm(getParamMain()->getDossier()+QString("Homol_Init"));
			if (!b) {
				setReaderror(QString("Homol_Init"));
				if (!*getAnnulation()) getParamMain()->setAvancement(-31);
				return;
			}
		}

		#if defined Q_WS_MAC
			//lien vers ElDcraw : dans le répertoire des données
			bool b = true;
			if (!QFile(getParamMain()->getDossier()+QString("bin")).exists()) {
				if (!QFile(getMicmacDir()+QString("bin")).link(getParamMain()->getDossier()+QString("bin"))) b = false;
			}
			if (!QFile(getParamMain()->getDossier()+QString("bin")).exists()) {
				QString commande = QString("ln -s %1bin %2bin").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier()));
				if (execute(commande)!=0) b = false;
			}
			if (!b) {
				//if (!*getAnnulation()) getParamMain()->setAvancement(-28);
				//return;
				cout << tr("Warning! Fail to link micmac/bin to data directory.").toStdString() << endl;
			}
		#endif
		//pour windows (Vista) :
		//mklink /D /H Lien Cible
		//lien vers ElDcraw : dans le répertoire de la console		
		/*QString consoleDir = QDir("./").absolutePath()+QString("/");	//repertoire de la console
		if (!QFile(consoleDir+QString("bin")).exists()) {
			bool b = QFile(getMicmacDir()+QString("bin")).link(consoleDir+QString("bin"));
			if (!b) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-28);
				return;
			}
		}*/

		#if defined Q_WS_WIN
			//pour make avec multi-processing
			if (!QFile(getParamMain()->getDossier()+QString("sh.exe")).exists() && QFile(getParamMain()->getMicmacDir()+QString("bin/sh.exe")).exists())
				QFile(getParamMain()->getMicmacDir()+QString("bin/sh.exe")).copy(getParamMain()->getDossier()+QString("sh.exe"));
		#endif

		//makefile pour la conversion des images au format tif (N&B et couleur)
		QString makefile = QString("MK")+getParamMain()->getDossier().section("/",-2,-2);
		getParamMain()->setMakeFile(makefile);
		QFile newFile(getParamMain()->getDossier()+makefile);
		if (!newFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
			getParamMain()->setAvancement(-100);
			setReaderror(conv(tr("for image conversion")));
			return;
		}
		QTextStream outStream(&newFile);
		QStringList outfiles;

		QString makefileJPG = QString("MK")+getParamMain()->getDossier().section("/",-2,-2)+QString("JPG");	//makefile pour les images jpg
		QFile newFileJPG(getParamMain()->getDossier()+makefileJPG);
		if (!newFileJPG.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
			getParamMain()->setAvancement(-100);
			setReaderror(conv(tr("for jpg image conversion")));
			return;
		}
		QTextStream outStreamJPG(&newFileJPG);

		//création du sous-dossier Images_brutes
		QDir dir(getParamMain()->getDossier());
		if (!QDir(getParamMain()->getDossier()+QString("Images_brutes")).exists()) {
			bool b = dir.mkdir("Images_brutes");
			if (!b) {
				getParamMain()->setAvancement(-4);
				setReaderror(QString("Images_brutes/"));
				return;
			}
			cout << tr("Sub-directory Images_brutes for raw images made.\n").toStdString();
		}

		for (int i=0; i<getParamMain()->getCorrespImgCalib().count(); i++) {
			QString imgRAW = getParamMain()->getCorrespImgCalib().at(i).getImageRAW();
			QString img = imgRAW.section(".",0,-2);
			QString imgTIF = img+QString(".tif");	//image tif (sans focale ajoutée si elle n'y est pas)
			QString imgTIF2 = getParamMain()->convertTifName2Couleur(imgTIF);	//image tif couleur (sans focale ajoutée si elle n'y est pas)
			getParamMain()->modifCorrespImgCalib()[i].setImageTif(imgTIF);

			//existance des images tif renommées ou non
			bool tifNB = false, tifClr = false;

			QStringList l = QDir(getParamMain()->getDossier()).entryList( QStringList(QString("F*_%1").arg(imgTIF)), QDir::Files);
			if (l.count()>0) {
				tifNB = true;
				getParamMain()->modifCorrespImgCalib()[i].setImageTif(l.at(0));
				imgTIF = l.at(0);
			} else if (QFile(getParamMain()->getDossier()+imgTIF).exists()) tifNB = true;

			QStringList l2 = QDir(getParamMain()->getDossier()).entryList( QStringList(QString("F*_%1").arg(imgTIF2)), QDir::Files);
			if (l2.count()>0 || QFile(getParamMain()->getDossier()+imgTIF2).exists()) tifClr = true;
			if (l2.count()>0) imgTIF2 = l2.at(0);

			if (tifClr && QFileInfo(getParamMain()->getDossier()+imgTIF2).isSymLink()) {
				deleteFile(getParamMain()->getDossier()+imgTIF2);
				tifClr = false;
			}

			//validité de l'image N&B (elle doit être lisible pour elise et pour Qt si l'image couleur n'existe pas et qu'il n'y a pas d'image RAW)
			if (tifNB) {		
				bool b = true;
				FILE * _fp = fopen((getParamMain()->getDossier()+imgTIF).toStdString().c_str(),"r");
				if (_fp!=0) {
					fclose( _fp );
					if (!tifClr
					&& imgRAW.section(".",-1,-1).toUpper()==QString("TIF") && imgRAW.section(".",-1,-1).toUpper()==QString("TIFF")
					&& QImage(getParamMain()->getDossier()+imgTIF).isNull())	//imgTIF sera aussi l'image couleur
						b = false;
				} else b = false;

				if (!b) {	//image non lisible
					cout << (QObject::tr("Unreadable image %1. Conversion in progress...").arg(noBlank(getParamMain()->getDossier()+imgTIF))).toStdString() << endl;
					QString img0 = imgTIF.section(".",0,-2);
					//conversion en jpg avec convert
					if (!QFile().exists(getParamMain()->getDossier()+img0+QString(".jpg"))) {
						QString convertCallName = QString( g_externalToolHandler.get("convert").callName().c_str() );
						QString commande = convertCallName+QString("convert %1%2.tif %1%2.jpg").arg(noBlank(getParamMain()->getDossier())).arg(img0);
						if (execute(commande)!=0) {
							cout << (QObject::tr("Fail to convert image %1 to JPG format.").arg(noBlank(getParamMain()->getDossier()+imgTIF)).toStdString()) << endl;
							getParamMain()->setAvancement(-35);
							setReaderror(imgTIF);
							return;
						}	//l'image JPG sera stockée dans le répertoire Images_Brutes ultérieurement
					}	
				
					//mise sur liste d'attente pour la reconversion en tif avec Devlop (même si le fichier n'a pas de métadonnées)
					QString commande = QString("%1bin/Devlop %2%3.jpg 8B=1 Gray=1 NameOut=%2%3.tif").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(img0);
					outStreamJPG << commande << "\n";					

					//déplacement de l'image tif illisible
					deleteFile(getParamMain()->getDossier()+QString("Images_brutes/")+imgTIF);	//il y a déjà une image de même nom (oubliée d'un autre calcul ?)
					QFile(getParamMain()->getDossier()+imgTIF).rename(getParamMain()->getDossier()+QString("Images_brutes/")+imgTIF);
				}
			}

			//validité de l'image couleur pour Qt
			if (tifClr) {
				if (QImage(getParamMain()->getDossier()+imgTIF2).isNull()) {	//image non lisible
					cout << (QObject::tr("Unreadable image %1. Conversion in progress...").arg(noBlank(getParamMain()->getDossier()+imgTIF2))).toStdString() << endl;
					QString img0 = imgTIF2.section(".",0,-2);
					//conversion en jpg avec convert
					if (!QFile(getParamMain()->getDossier()+img0+QString(".jpg")).exists()) {
						QString commande = QString("convert %1%2.tif %1%2.jpg").arg(noBlank(getParamMain()->getDossier())).arg(img0);
						if (execute(commande)!=0) {
							cout << (QObject::tr("Fail to convert image %1 to JPG format.").arg(noBlank(getParamMain()->getDossier()+imgTIF2))).toStdString() << endl;
							getParamMain()->setAvancement(-35);
							setReaderror(imgTIF2);
							return;
						}	//l'image JPG sera stockée dans le répertoire Images_Brutes ultérieurement
					}

					//mise sur liste d'attente pour la reconversion en tif avec Devlop (même si le fichier n'a pas de métadonnées)
					QString commande = QString("%1bin/Devlop %2%3.jpg Gray=0 NameOut=%2%3.tif").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(img0);
					outStreamJPG << commande << "\n";	

					//déplacement de l'image tif illisible
					deleteFile(getParamMain()->getDossier()+QString("Images_brutes/")+imgTIF2);	//il y a déjà une image de même nom (oubliée d'un autre calcul ?)
					QFile(getParamMain()->getDossier()+imgTIF2).rename(getParamMain()->getDossier()+QString("Images_brutes/")+imgTIF2);
				}
			}
			if (tifNB && (tifClr || (imgRAW.section(".",-1,-1).toUpper()==QString("TIF") && imgRAW.section(".",-1,-1).toUpper()==QString("TIFF")))) continue;

			//écriture des conversions
			if (imgRAW.section(".",-1,-1).toUpper()==QString("JPG") || imgRAW.section(".",-1,-1).toUpper()==QString("JPG")) {	//jpg -> traitée à part
				if (!tifNB) {
					QString commande = QString("%1bin/Devlop %2%3 8B=1 Gray=1 NameOut=%2%4").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(imgRAW).arg(imgTIF);
					outStreamJPG << commande << "\n";
				}

				if (!tifClr) {
					QString commande2 = QString("%1bin/Devlop %2%3 Gray=0 NameOut=%2%4").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(imgRAW).arg(imgTIF2);
					outStreamJPG << commande2 << "\n";
				}
			} else if (imgRAW.section(".",-1,-1).toUpper()!=QString("TIFF") && imgRAW.section(".",-1,-1).toUpper()!=QString("TIF")) {	//raw
				if (!tifNB) {
					#if defined Q_WS_MAC	//problème de lien vers bin
						QString commande = QString("cd %1").arg(noBlank(getMicmacDir()));
						QString commande2 = QString("%1bin/MpDcraw %1 %2 16B=0 GB=1").arg(noBlank(getParamMain()->getDossier())).arg(imgRAW).arg(noBlank(getMicmacDir()));	//->img_MpDcraw8B_GB.tif
						QString commande3 = QString("mv %1%2_MpDcraw8B_GB.tif %1%3").arg(noBlank(getParamMain()->getDossier())).arg(img).arg(imgTIF);
						outStream << imgTIF << " :\n\t" << commande << "\n";
						outStream << "\t" << commande2 << "\n";
						outStream << "\t" << commande3 << "\n";
						outfiles.push_back(imgTIF);
					#else
						QString commande = comm(QString("%1bin/MpDcraw %2 %3 16B=0 GB=1").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(imgRAW));
						QString commande2 = comm(QString("mv %1%2_MpDcraw8B_GB.tif %1%3").arg(noBlank(getParamMain()->getDossier())).arg(img).arg(imgTIF));
						outStream << imgTIF << " :\n\t" << commande << "\n";
						outStream << "\t" << commande2 << "\n";
						outfiles.push_back(imgTIF);
					#endif
				}

				if (!tifClr) {
					QString commande4 = comm(QString("%1bin/ElDcraw -T -t 0 -w %2%3").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(imgRAW));	//->img.tiff
					QString tempofile = imgRAW.section(".",0,-2)+QString(".tiff");
					#if defined Q_WS_WIN
						QString commande5 = comm(QString("move %1%2 %1%3").arg(comm(noBlank(getParamMain()->getDossier()))).arg(tempofile).arg(imgTIF2));
					#else
						QString commande5 = QString("mv %1%2 %1%3").arg(noBlank(getParamMain()->getDossier())).arg(tempofile).arg(imgTIF2);
					#endif
					outStream << imgTIF2 << " :\n\t" << commande4 << "\n";
					outStream << "\t" << commande5 << "\n";
					outfiles.push_back(imgTIF2);	//! tiff_info ne donne pas la focale de cette image
				}
			}		
		}
		outStream << "all :";
		for (int i=0; i<outfiles.count(); i++)
			outStream << " " << outfiles.at(i);
		newFile.close();
		newFileJPG.close();

		getParamMain()->setAvancement(Conversion);
		cout << tr("Makefile for image conversion made.").toStdString() << endl;
	}
	
	if (getParamMain()->getAvancement()==Conversion) {
		emit saveCalcul();	
		if (*getAnnulation()) return;
		setProgressLabel(tr("Image conversion"));

		//images raw
		//QString commande = comm(QString("cd %1 & %5make all -f %2%3 -j%4").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(noBlank(getParamMain()->getMakeFile())).arg(cpu_count).arg(dirBin(getParamMain()->getMicmacDir())));
		QString commande = comm(QString("cd %1 & %5 all -f %2%3 -j%4").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(noBlank(getParamMain()->getMakeFile())).arg(cpu_count).arg( (QString)( g_externalToolHandler.get("make").callName().c_str() ) ));
		if (execute(commande)!=0) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-3);
			return;
		}
		cout << tr("Raw images converted into tif format.").toStdString() << endl;


		//images jpg
		/*bool b = false;
		for (int i=0; i<getParamMain()->getCorrespImgCalib().count(); i++) {
			QString extension = getParamMain()->getCorrespImgCalib().at(i).getImageRAW().section(".",-1,-1).toUpper();
			if (extension==QString("JPG") || extension==QString("JPEG")) {
				b = true;
				break;
			}
		}*/
		QFile makefileJPG(getParamMain()->getDossier()+getParamMain()->getMakeFile()+QString("JPG"));
		if (!makefileJPG.open(QIODevice::ReadOnly | QIODevice::Text)) {
			cout << (tr("Fail to read file %1.").arg(getParamMain()->getDossier()+getParamMain()->getMakeFile()+QString("JPG"))).toStdString() << endl;
			if (!*getAnnulation()) getParamMain()->setAvancement(-3);
			return;
		}
		QTextStream instreamJPG(&makefileJPG);
		while (!instreamJPG.atEnd()) {
			QString text = instreamJPG.readLine().simplified().trimmed();
			if (text.isEmpty()) continue;
			QString imgOut = text.section(" ",-1,-1).section("=",-1,-1);
			if (QFile(imgOut).exists()) continue;
			if (execute(text)==0) continue;
			QString imgIn = text.section(" ",1,1).section("/",-1,-1);
			if (!QFile(getParamMain()->getDossier()+QString("TmpConvert_XJI_%1.tif").arg(imgIn.section(".",0,-2))).rename(imgOut)) {
				cout << tr("Fail to convert image %1 into tif format.").arg(imgIn).toStdString() << endl;
				if (!*getAnnulation()) getParamMain()->setAvancement(-3);
				return;
			}
			QFile(getParamMain()->getDossier()+imgIn).rename(getParamMain()->getDossier()+QString("Images_brutes/")+imgIn);
		}
		makefileJPG.close();

			//nettoyage
		QStringList l = QDir(QApplication::applicationDirPath()).entryList( QStringList(QString("XXX-hkjyur-toto*.txt")), QDir::Files);
		if (l.count()>0) {
			for (QStringList::const_iterator it=l.begin(); it!=l.end(); it++)
				QFile(QApplication::applicationDirPath()+QString("/")+*it).remove();
		}
		cout << tr("Jpg images converted into tif format.").toStdString() << endl;
		getParamMain()->setAvancement(Ecriture);
	}
	
	if (getParamMain()->getAvancement()==Ecriture) {
		emit saveCalcul();	
		if (*getAnnulation())
		if (*getAnnulation()) return;
		setProgressLabel(conv(tr("Image moving")));

		//création du sous-dossier Images_brutes
		QDir dir(getParamMain()->getDossier());
		if (!QDir(getParamMain()->getDossier()+QString("Images_brutes")).exists()) {
			bool b = dir.mkdir("Images_brutes");
			if (!b) {
				getParamMain()->setAvancement(-4);
				setReaderror(QString("Images_brutes/"));
				return;
			}
			cout << tr("Sub-directory Images_brutes for raw images made.\n").toStdString();
		}

		//organisation des images
		for (int i=0; i<getParamMain()->getCorrespImgCalib().count(); i++) {
			QString imageRAW = getParamMain()->getCorrespImgCalib().at(i).getImageRAW();
			QString extension = imageRAW.section(".",-1,-1);
				//déplacement des images raw et jpg
			if (extension.toUpper()!=QString("TIFF") && extension.toUpper()!=QString("TIF")) {	//image raw ou jpg -> à déplacer
				if (QFile(getParamMain()->getDossier()+imageRAW).exists())
					QFile(getParamMain()->getDossier()+imageRAW).rename(getParamMain()->getDossier()+QString("Images_brutes/")+imageRAW);
			} else {
				//renomination des images tif
				QString imgTif = getParamMain()->getCorrespImgCalib().at(i).getImageTif();
				if (imageRAW!=imgTif && QFile(getParamMain()->getDossier()+imageRAW).exists())	//à renommer
					QFile(getParamMain()->getDossier()+imageRAW).rename(getParamMain()->getDossier()+imgTif);
			}
		}
		cout << tr("Raw and jpg images are moved. Tif images are renamed (extension).").toStdString() << endl;

		//associations des images et des calibrations
		QList<int> calibDone; 	//calibrations traitées, afin de supprimer celles en trop
		QVector<int> focales(getParamMain()->getCorrespImgCalib().count());	//focales des images
		for (int i=0; i<getParamMain()->getCorrespImgCalib().count(); i++) {
			QString imgTif = getParamMain()->getCorrespImgCalib().at(i).getImageTif();
			//focale et taille de l'image
			int f(0);	
			QSize taille(0,0);
			if (!focaleTif(getParamMain()->getDossier()+imgTif, getMicmacDir(), &f, &taille)
			&& !focaleTif(getParamMain()->getDossier()+imgTif, getMicmacDir(), 0, &taille)) {	//impossible de lire la taille de l'image avec tiff_info
				QImage imageQt(getParamMain()->getDossier()+imgTif);
				if (!imageQt.isNull())	//test récupération de la taille avec Qt
					taille = imageQt.size();
				else {
					imageQt = QImage(getParamMain()->getDossier()+getParamMain()->convertTifName2Couleur(imgTif));
					if (!imageQt.isNull())	//on teste aussi l'image couleur (sinon elle sera supprimée car non lisible par Qt)
						taille = imageQt.size();
					else {
						getParamMain()->setAvancement(-35);
						setReaderror(imgTif);
						return;
					}
				}
			}
			getParamMain()->modifCorrespImgCalib()[i].setTaille(taille);
			if (f==0) {
				if (getParamMain()->getParamPastis().getCalibFiles().count()==1) f = getParamMain()->getParamPastis().getCalibFiles().at(0).second;
				else {
					bool ok = false;
					QString fstr = imgTif;
					fstr = fstr.left(4).right(3);	
					f = fstr.toInt(&ok);
					if (!ok) {
						f = 0;
						getParamMain()->setAvancement(-35);
						setReaderror(imgTif);
						return;
					}
					fstr = QVariant(f).toString();
					while (fstr.count()<3) fstr = QString("0") + fstr;
					if (getParamMain()->getCorrespImgCalib().at(i).getImageTif().left(5)!=QString("F%1_").arg(fstr)) {
						f = 0;
						getParamMain()->setAvancement(-35);
						setReaderror(imgTif);
						return;
					}
				}
			}
			focales[i] = f;

			//association avec les calibrations
			int j = 0;
			while (j<getParamMain()->getParamPastis().getCalibFiles().count()) {
				if (getParamMain()->getParamPastis().getCalibFiles().at(j).second!=f) {
					j++;
					continue;
				}
				getParamMain()->modifCorrespImgCalib()[i].setCalibration(getParamMain()->getParamPastis().getCalibFiles().at(j).first);
				if (calibDone.contains(j)) break;
				//remplissage des paramètres par défaut de la calibration à partir de la taille de l'image
                        	CalibCam calib;
				QString error(FichierCalibCam::lire(getParamMain()->getDossier(), getParamMain()->getCorrespImgCalib().at(i).getCalibration(), calib));
				if (!error.isEmpty()) {
					getParamMain()->setAvancement(-11);
					setReaderror(getParamMain()->getCorrespImgCalib().at(i).getCalibration() + QString(" : ") + error);
					return;
				}
				if (!calib.setDefaultParam(taille)){
					getParamMain()->setAvancement(-13);
					return;
				}
				if (!FichierCalibCam::ecrire(getParamMain()->getDossier(), calib)){
					getParamMain()->setAvancement(-12);
					return;
				}
				calibDone.push_back(j);
				break;
			}
			if (j==getParamMain()->getParamPastis().getCalibFiles().count()) {	//pas de calibration pour cette image
				getParamMain()->setAvancement(-17);
				return;
			}
		}
		//suppression des calibrations en trop
		for (int i=0; i<getParamMain()->getParamPastis().getCalibFiles().count(); i++) {
			if (!calibDone.contains(i)) {
				getParamMain()->modifParamPastis().modifCalibFiles().removeAt(i);
				i--;
			}
		}
		cout << tr("Calibration parameters completed.").toStdString() << endl;

		//ajout de la focale au nom des images tif
		for (int i=0; i<getParamMain()->getCorrespImgCalib().count(); i++) {
			QString imgTif = getParamMain()->getCorrespImgCalib().at(i).getImageTif();
			QString f = QVariant(focales.at(i)).toString();
			while (f.count()<3) f = QString("0") + f;
			if (f.count()>3) {
				getParamMain()->setAvancement(-36);
				setReaderror(imgTif);
				return;
			}
			if (imgTif.left(5)!=QString("F%1_").arg(f)) {	//image non déjà renommée
				QString imgTif2 = QString("F%1_%2").arg(f).arg(imgTif);
				if (!QFile(getParamMain()->getDossier()+imgTif2).exists()) QFile(getParamMain()->getDossier()+imgTif).rename(getParamMain()->getDossier()+imgTif2);
				imgTif = imgTif2;
			}

			//images tif couleur (lien)
			QString imgTifCouleur = getParamMain()->convertTifName2Couleur(imgTif.right(imgTif.count()-5));
			QString imgTifCouleur2 = QString("F%1_%2").arg(f).arg(imgTifCouleur);
			if (!QFile(getParamMain()->getDossier()+imgTifCouleur).exists() && !QFile(getParamMain()->getDossier()+imgTifCouleur2).exists()) {	//création d'un lien
		
#if ELISE_windows
				// Cannot create a symbolic link under windows so copy the file
				bool b = QFile(getParamMain()->getDossier()+imgTif).copy(getParamMain()->getDossier()+imgTifCouleur2);
#else
				bool b = QFile(getParamMain()->getDossier()+imgTif).link(getParamMain()->getDossier()+imgTifCouleur2);		
#endif
				if (!b || !QFile(getParamMain()->getDossier()+imgTifCouleur2).exists()) {
#if ELISE_windows
					cerr << "unable to copy " << ( getParamMain()->getDossier()+imgTif ).toStdString() << endl;
#else
					QString commande = QString("ln -s %1%2 %1%3").arg(getParamMain()->getDossier()).arg(imgTif).arg(imgTifCouleur2);
					if (execute(commande)!=0) {
						if (!*getAnnulation()) getParamMain()->setAvancement(-33);
						setReaderror(imgTif);
						return;
					}
#endif
				}
			}
			else if (!QFile(getParamMain()->getDossier()+imgTifCouleur2).exists())	//renomination
				QFile(getParamMain()->getDossier()+imgTifCouleur).rename(getParamMain()->getDossier()+imgTifCouleur2);
			getParamMain()->modifCorrespImgCalib()[i].setImageTif(imgTif);
		}

		//identifiant des images
		if (!getParamMain()->calcImgsId()) {
			getParamMain()->setAvancement(-34);
			return;
		}

		//écriture de la liste d'images tif dans le fichier xml
		if (!FichierParamImage::ecrire(getParamMain()->getDossier()+getParamMain()->getImageXML(), getParamMain()->getCorrespImgCalib())) {
			getParamMain()->setAvancement(-6);
			return;
		}
		cout << tr("Image list saved.").toStdString() << endl;

		//écriture de la liste des couples d'images
		QString err = FichierCouples::convertir (getParamMain()->getDossier()+getParamMain()->getCoupleXML(), getParamMain()->getCorrespImgCalib());
		if (!err.isEmpty()) {
			setReaderror(err);
			getParamMain()->setAvancement(-7);
			return;
		}
		cout << tr("Image pairs are updated.").toStdString() << endl;

		//écriture du fichier des associations image - calibration interne
		if (!FichierAssocCalib::ecrire (getParamMain()->getDossier()+getParamMain()->getAssocCalibXML(), getParamMain()->getCorrespImgCalib())) {
			getParamMain()->setAvancement(-8);
			return;
		}
		cout << tr("Image - calibration matches saved.").toStdString() << endl;

		//écriture du fichier .xml
		/*QString patronXml;
		switch (getParamMain()->getParamPastis().getTypeChantier()) {
			case ParamPastis::Convergent :
				patronXml = "ChantierDescrConver.xml";
				break;
			case ParamPastis::Bandes :
				patronXml = "ChantierDescrBandes.xml";
				break;
			case ParamPastis::Autre :
				patronXml = "ChantierDescrAutre.xml";
				break;
		}*/
		QString patronXml = "ChantierDescrConver.xml";
		QFile file (getParamMain()->getMicmacDir()+"/interface/xml/"+patronXml);
		deleteFile(getParamMain()->getDossier() + getParamMain()->getChantierXML());
		file.copy(getParamMain()->getDossier() + getParamMain()->getChantierXML());

		cout << tr("Default survey parameters loaded.").toStdString() << endl;
		
		//calcul du makefile des points d'intérêt
		getParamMain()->setMakeFile(QString("MK")+getParamMain()->getDossier().section("/",-2,-2)+QString("b"));
		
		QString commande = comm(noBlank(getMicmacDir()) + QString("bin/Pastis ") + noBlank(getParamMain()->getDossier()) + QString(" Key-Rel-All-Cple %1").arg(getParamMain()->getParamPastis().getLargeurMax()));
		commande += comm(QString(" FiltreOnlyDupl=1 MkF=") + noBlank(getParamMain()->getDossier()) + noBlank(getParamMain()->getMakeFile()) + QString("  NbMinPtsExp=2"));
		if (execute(commande)!=0) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-9);
			return;
		}
		cout << tr("Makefile for Pastis made").toStdString() << endl;
		getParamMain()->modifParamPastis().modifCouples().clear();
		QString lecture = FichierCouples::lire (getParamMain()->getDossier()+getParamMain()->getCoupleXML(), getParamMain()->modifParamPastis().modifCouples(), getParamMain()->getCorrespImgCalib());
		if (!lecture.isEmpty()) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-26);
			return;
		}
		cout << tr("image pairs read").toStdString() << endl;
		if (!getParamMain()->getParamPastis().getMultiScale()) getParamMain()->setAvancement(PtsInteret);
		else getParamMain()->setAvancement(FiltrCpls);
	}
	
	if (getParamMain()->getAvancement()==FiltrCpls) {
		if (*getAnnulation()) {
			return;
		}
		setProgressLabel(conv(tr("Point of interest computing (first step)")));

		//cas où le nom du dossier (ou d'un de ses parents) est composé
		if (!corrigeMakefile()) return;

		QString commande = QString("cd ") + comm(noBlank(getMicmacDir())) + QString(" & ");
		QString makeCallName( g_externalToolHandler.get("make").callName().c_str() );
		commande += makeCallName + QString(" all -f ") + noBlank(getParamMain()->getDossier()) + noBlank(getParamMain()->getMakeFile()) + QString(" -j%1").arg(cpu_count) + QString(" >")+noBlank(getStdoutfile());
		if (execute(commande)!=0) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-10);
			return;
		}
		cout << tr("Tie-point computation first step done.").toStdString() << endl;

		//tri des couples
		cTplValGesInit<string>  aTpl;
		char** argv = new char*[1];
		char c_str[] = "rthsrth";
		argv[0] = new char[strlen( c_str )+1];
		strcpy( argv[0], c_str );
		getParamMain()->modifParamPastis().modifCouples().clear();
		cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, getParamMain()->getDossier().toStdString(), aTpl );
		const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");
		for (int aK=0; aK<signed(aVN->size()) ; aK++) {
			//bon couple
			pair<string,string>  aPair = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);
			//extraction des points
			ElPackHomologue aPack = ElPackHomologue::FromFile(getParamMain()->getDossier().toStdString()+(*aVN)[aK]);
			if (aPack.size()<getParamMain()->getParamPastis().getNbPtMin()) continue;
			if (!getParamMain()->getParamPastis().getCouples().contains( pair<QString, QString>( QString(aPair.second.c_str()) , QString(aPair.first.c_str()) ) ))
				getParamMain()->modifParamPastis().modifCouples().push_back( pair<QString, QString>( QString(aPair.first.c_str()) , QString(aPair.second.c_str()) ) );
		}
		delete [] argv[0];
		delete [] argv;
		delete mICNM;

		//écriture du fichier des couples
		QFile(getParamMain()->getDossier()+getParamMain()->getCoupleXML()).rename(getParamMain()->getDossier()+getParamMain()->getCoupleXML().section(".",0,-2)+QString("_init.xml"));
		if (!FichierCouples::ecrire (getParamMain()->getDossier()+getParamMain()->getCoupleXML(), getParamMain()->getParamPastis().getCouples())) {
			getParamMain()->setAvancement(-25);
			return;
		}
		QDir(getParamMain()->getDossier()).rename(QString("Homol"), QString("Homol_Init"));
		QDir(getParamMain()->getDossier()).rename(QString("Pastis"), QString("Pastis_Init"));

		//makefile pour la 2ème passe
		getParamMain()->setMakeFile(QString("MK")+getParamMain()->getDossier().section("/",-2,-2)+QString("b"));
		//QString newTempoDir = QString("cd ") + noBlank(getParamMain()->getDossier()) + QString("\n");
		commande = noBlank(getMicmacDir()) + QString("bin/Pastis ") + noBlank(getParamMain()->getDossier()) + QString(" Key-Rel-All-Cple %1").arg(getParamMain()->getParamPastis().getLargeurMax2());
		commande += QString(" FiltreOnlyDupl=1 MkF=") + noBlank(getParamMain()->getDossier()) + noBlank(getParamMain()->getMakeFile()) + QString("  NbMinPtsExp=2");
		if (execute(commande)!=0) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-9);
			return;
		}
		cout << tr("Makefile for Pastis made (second step)").toStdString() << endl;

		getParamMain()->setAvancement(PtsInteret);
	}
	
	if (getParamMain()->getAvancement()==PtsInteret) {
		if (*getAnnulation()) {
			return;
		}
		setProgressLabel(conv(tr("Point of interest computing")));

		//cas où le nom du dossier (ou d'un de ses parents) est composé
		if (!corrigeMakefile()) return;

		QString commande;
		#if defined Q_WS_WIN
			commande = QString("cd ") + comm(noBlank(getMicmacDir()+QString("bin/"))) + QString(" & ");	//pour utiliser sh.exe
		#else
			commande = QString("cd ") + comm(noBlank(getMicmacDir())) + QString(" & ");	//pour avoir des chemins relatifs
		#endif
		QString makeCallName( g_externalToolHandler.get("make").callName().c_str() );
		commande += makeCallName + QString(" all -f ") + noBlank(getParamMain()->getDossier()) + noBlank(getParamMain()->getMakeFile()) + QString(" -j%1").arg(cpu_count) + QString(" >")+noBlank(getStdoutfile());
		if (execute(commande)!=0) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-10);
			return;
		}
		getParamMain()->modifParamApero().setFiltrage(true);
		cout << tr("Tie-points computed.").toStdString() << endl;
		getParamMain()->setAvancement(ImgsOrientables);
	}
	
	if (getParamMain()->getAvancement()==ImgsOrientables) {
		if (*getAnnulation()) {
			return;
		}

		QString err = defImgOrientables();
		if (!err.isEmpty())  {
			if (!*getAnnulation()) getParamMain()->setAvancement(-22);
			setReaderror(err);
			return;
		}
		cout << tr("Computable poses defined.").toStdString() << endl;
		getParamMain()->setAvancement(Termine);
		cout << tr("Done").toStdString() << endl;
	}

	setProgressLabel(conv(tr("Done")));
}

bool PastisThread::corrigeMakefile() {
	bool b = false;
	#if (defined Q_WS_X11)
		b = true;
	#endif
	/*bool b2 = false;
	#if (defined Q_WS_WIN)
		b2 = true;
	#endif*/
	if (getParamMain()->getDossier().contains(QString(" ")) || getParamMain()->getMicmacDir().contains(QString(" ")) || b/* || b2*/) {
		QDir consoleDir = QDir::current();	
		QString newMicmacDir = b ? noBlank(consoleDir.relativeFilePath(getParamMain()->getMicmacDir())+QString("/")) : noBlank(getParamMain()->getMicmacDir());
		QString newDossier = b ? noBlank(consoleDir.relativeFilePath(getParamMain()->getDossier())+QString("/")) : noBlank(getParamMain()->getDossier());
		
		QFile oldFile(getParamMain()->getDossier() + getParamMain()->getMakeFile());
		if (!oldFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
			getParamMain()->setAvancement(-16);
			return false;
		}
		QTextStream inStream(&oldFile);
		QFile newFile(getParamMain()->getDossier() + "tempo");
		if (!newFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
			getParamMain()->setAvancement(-100);
			setReaderror(QApplication::translate("Dialog", "for tie-point search", 0, QApplication::CodecForTr));
			return false;
		}
		QTextStream outStream(&newFile);
		QString text = inStream.readAll();
		text.replace(getParamMain()->getMicmacDir(), newMicmacDir);
		text.replace(getParamMain()->getDossier(), newDossier);
		text.replace(getParamMain()->getMakeFile(), noBlank(getParamMain()->getMakeFile()));
		/*if (b2) {
			text.replace("LBPpPastis/", "LBPp");
			text.replace("LBPp-Match-Pastis/", "LBPp-Match-");
			text.replace(".tif/Pastis/", ".tif/");
		}*/
		outStream << text;			
		oldFile.close();
		newFile.close();
		oldFile.remove();
		newFile.rename(getParamMain()->getDossier() + getParamMain()->getMakeFile());
	}

	#if defined Q_WS_WIN
		QDir(getParamMain()->getDossier()+QString("Pastis/")).mkdir("LBPpPastis/");
		QString Pdir = getParamMain()->getDossier()+QString("Pastis/LBPp-Match-Pastis/LBPpPastis/");
		QStringList l = QDir(Pdir).entryList(QDir::Dirs);
		for (QStringList::const_iterator it=l.begin(); it!=l.end(); it++) {
			if (it->left(1)==QString(".")) continue;
			QDir(Pdir+*it).mkdir("Pastis");
			QDir(Pdir+*it+QString("/Pastis/")).mkdir("LBPpPastis");
		}
		for (QVector<ParamImage>::const_iterator it=getParamMain()->getCorrespImgCalib().begin(); it!=getParamMain()->getCorrespImgCalib().end(); it++)
			QDir(getParamMain()->getDossier()+QString("Homol")).mkdir(QString("Pastis")+it->getImageTif().section(".",0,-2));
	#endif
	return true;
}

QString PastisThread::defImgOrientables() {
	//on exclut des images initiales les images qui ne sont pas liées au chantier (pas de couple correspondant dans Homol)
//(en fait Apero les sélectionne différemment, mais ça permet de faire un pré-filtrage et évite à l'application de planter)

	//lecture des couples d'images
	cTplValGesInit<string>  aTpl;
	char** argv = new char*[1];
	char c_str[] = "rthsrth";
	argv[0] = new char[strlen( c_str )+1];
	strcpy( argv[0], c_str );
	QList<pair<QString, QString> >* couples = new QList<std::pair<QString, QString> >;
	cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, getParamMain()->getDossier().toStdString(), aTpl );
	const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");
	for (int aK=0; aK<signed(aVN->size()) ; aK++) {
		//bon couple
		pair<string,string>  aPair = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);
		//extraction des points
		ElPackHomologue aPack = ElPackHomologue::FromFile(getParamMain()->getDossier().toStdString()+(*aVN)[aK]);
		if (aPack.size()==0) continue;
		couples->push_back( pair<QString, QString>( QString(aPair.first.c_str()) , QString(aPair.second.c_str()) ) );
	}
	delete [] argv[0];
	delete [] argv;
	delete mICNM;

	//calcul des composantes connexes
		//initialisation
	QList<QStringList > composantes;
	for (QVector<ParamImage>::const_iterator it=getParamMain()->getCorrespImgCalib().begin(); it!=getParamMain()->getCorrespImgCalib().end(); it++) {
		QStringList l;
		l.push_back(it->getImageTif());
		composantes.push_back(l);
	}
		//rattachement
	for (QList<pair<QString, QString> >::iterator it=couples->begin(); it!=couples->end(); it++) {
		//recherche de l'img1
		int l1 = -1;
		int l2 = -1;
		for (int i=0; i<composantes.count(); i++) {
			if (composantes.at(i).indexOf(it->first)!=-1) {
				l1 = i;
				break;
			}
		}
		if ( l1==-1 ) continue;
		//recherche de l'img2
		for (int i=0; i<composantes.count(); i++) {
			if (composantes.at(i).indexOf(it->second)!=-1) {
				l2 = i;
				break;
			}
		}
		if ( (l2==-1) || (l1==l2) ) continue;
		//transfert
		composantes[l1].append(composantes.at(l2));
		composantes.removeAt(l2);
	}

	//enregistrement du nouveau chantier
	getParamMain()->modifParamApero().modifImgToOri().clear();
	getParamMain()->modifParamApero().modifImgToOri() = *(composantes.begin());
	for (int i=0; i<composantes.count(); i++) {	//on prend la plus grosse composante
		if (composantes.at(i).count()>getParamMain()->getParamApero().getImgToOri().count())
			getParamMain()->modifParamApero().modifImgToOri() = composantes.at(i);
	}	// (si plusieurs composantes, exécution en plusieurs parties ?)
	//if (imgOri->count()<3) return tr("Il y a moins de 3 images à orienter après vérification des points homologues.");

	return QString();
}

bool PastisThread::killProcess() {
	bool b = false;
	if (killall("Devlop")==0) b = true;
	if (killall("tiff_info")==0) b = true;
	if (killall("Tapioca")==0) b = true;
	if (killall("MapCmd")==0) b = true;
	if (killall("MpDcraw")==0) b = true;
	#if defined Q_WS_WIN
		if (killall("move")==0) b = true;
	#else
		if (killall("mv")==0) b = true;
	#endif
	if (killall("Pastis")==0) b = true;
	#if defined Q_WS_MAC
		if (killall("siftpp_tgi.OSX")==0) b = true;
		if (killall("ann_samplekey200filtre.OSX")==0) b = true;
	#elif defined Q_OS_LINUX
		if (killall("siftpp_tgi.LINUX")==0) b = true;
		if (killall("ann_mec_filtre.LINUX")==0) b = true;
	#elif (defined Q_WS_WIN || defined Q_WS_X11)
		if (killall("siftpp_tgi.exe")==0) b = true;
		if (killall("ann_samplekeyfiltre.exe")==0) b = true;
	#endif
	if (killall("ElDcraw")==0) b = true;
	if (killall("tiff2rgba")==0) b = true;
	return b;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*	auto-calibration : AperoConvAutoCalib.xml -> Apero0.xml = Ori-Finale
	paramètres minimaux : 1 étape = AperoConver.xml -> Apero.xml = Ori-Finale
	multi-échelle : 2 étapes = AperoConvCourteFoc.xml -> Apero.xml + AperoConvLongFoc.xml -> Apero2.xml = Orient + Ori-Finale
	orientation manuelle : dans la 1ière étape = + OrientationPlanDir.xml
	dissociation des calibrations : + 1 étape = AperoLiberCalib.xml -> Apero3.xml = Ori-Finale-3
*/

AperoThread::AperoThread (ParamMain* pMain, const QString& stdoutfilename, bool* annul, int cpu) : AppliThread(pMain,stdoutfilename,annul), zoneChantier(QVector<GLdouble>(6)), zoneChantierEtCam(QVector<GLdouble>(6)), cpu_count(cpu)
	{ setEndResult(Termine); }
AperoThread::AperoThread(const AperoThread& aperoThread) : AppliThread() { copie(this,aperoThread); }
AperoThread::~AperoThread () {}

AperoThread& AperoThread::operator=(const AperoThread& aperoThread) {
	if (this!=&aperoThread) copie(this,aperoThread);
	return *this;
}

void copie(AperoThread* aperoThread1, const AperoThread& aperoThread2) {
	copie(dynamic_cast<AppliThread*>(aperoThread1), aperoThread2);
	aperoThread1->echelle = aperoThread2.echelle;
	aperoThread1->zoneChantier = aperoThread2.zoneChantier;
	aperoThread1->zoneChantierEtCam = aperoThread2.zoneChantierEtCam;
	aperoThread1->cpu_count = aperoThread2.cpu_count;
}

void AperoThread::run() {

	if (getParamMain()->getAvancement()<=Enregistrement) {	
		setProgressLabel(conv(tr("User data saving")));

		//orientation initiale
		if (getParamMain()->getParamApero().getUseOriInit() && getParamMain()->getParamApero().getDirOriInit()!=QString("Ori-Initiale")) {
			//sauvegarde des autres orientations
			if (QDir(getParamMain()->getDossier()+QString("Ori-Initiale")).exists()) {
				int i = 1;
				while (QDir(getParamMain()->getDossier()+QString("Ori-Initiale%1").arg(i)).exists()) i++;
				QDir(getParamMain()->getDossier()).rename(QString("Ori-Initiale"),QString("Ori-Initiale%1").arg(i));
			}

			//renomination des fichiers-> à faire avant l'effacement des fichiers précédents au cas où
			QDir(getParamMain()->getDossier()).rename(getParamMain()->getParamApero().getDirOriInit(),QString("Ori-Initiale"));
			QStringList l = QDir(getParamMain()->getDossier()+QString("Ori-Initiale")).entryList(QDir::Files);
			for (QStringList::const_iterator it=l.begin(); it!=l.end(); it++) {
				QString imgName;
				for (QVector<ParamImage>::const_iterator it2=getParamMain()->getCorrespImgCalib().begin(); it2!=getParamMain()->getCorrespImgCalib().end(); it2++) {
					if (it->contains(it2->getImageTif().section(".",0,-2))) {
						imgName = it2->getImageTif().section(".",0,-2);
						break;
					}
				}
				
				if (!imgName.isEmpty()) QFile(getParamMain()->getDossier()+QString("Ori-Initiale/")+*it).rename(getParamMain()->getDossier()+QString("Ori-Initiale//OrInitiale-%1.xml").arg(imgName));
			}
			getParamMain()->modifParamApero().setDirOriInit(QString("Ori-Initiale"));
		}

		//effacement des résultats précédents
		QStringList dossiers;
		dossiers.push_back("Ori-AutoCalib");
		dossiers.push_back("Orient");
		dossiers.push_back("Ori-F");
		dossiers.push_back("Ori-Interm");
		dossiers.push_back("Ori-F-2");
		dossiers.push_back("Ori-F-3");
		for (int i=0; i<dossiers.count(); i++) {
			if (QDir(getParamMain()->getDossier()+dossiers.at(i)).exists()) {
				bool b = rm(getParamMain()->getDossier()+dossiers.at(i));
				if (!b) {
					setReaderror(dossiers.at(i));
					getParamMain()->setAvancement(-14);
					return;
				}
			}
		}

		//pour l'auto-calibration : tri des images par objectif
		QList<pair<QStringList,QString> > imagesTriees;
		QVector<int> numFocales(0);
		trieImgAutoCalib(imagesTriees, numFocales);

		//copie du fichier apero.xml
		if (getParamMain()->getParamApero().getAutoCalib().count()>0) {
			QString newFile(getParamMain()->getDossier() + getParamMain()->getAperoXML().section(".",0,-2) + QString("0.xml"));
			deleteFile(newFile);
			QFile file (getParamMain()->getMicmacDir()+"/interface/xml/AperoConvAutoCalib.xml");
			file.copy(newFile);
		}
		if (!getParamMain()->getParamApero().getMultiechelle()) {
			QString patronXml;
			/*switch (getParamMain()->getParamPastis().getTypeChantier()) {
				case ParamPastis::Convergent :
					patronXml = "AperoConver.xml";
					break;
				case ParamPastis::Bandes :
					patronXml = "AperoBandes.xml";
					break;
				case ParamPastis::Autre :
					patronXml = "AperoAutre.xml";
					break;
			}*/
			patronXml = "AperoConver.xml";
			QString newFile(getParamMain()->getDossier() + getParamMain()->getAperoXML());
			deleteFile(newFile);
			QFile file (getParamMain()->getMicmacDir()+"/interface/xml/"+patronXml);
			file.copy(newFile);
		} else {
			QString newFile1(getParamMain()->getDossier() + getParamMain()->getAperoXML());
			deleteFile(newFile1);
			QFile file( getParamMain()->getMicmacDir()+"/interface/xml/AperoConvCourteFoc.xml" );
			file.copy(newFile1);

			QString newFile2(getParamMain()->getDossier() + getParamMain()->getAperoXML().section(".",-0,-2)+QString("2.xml"));
			deleteFile(newFile2);
			QFile file2( getParamMain()->getMicmacDir()+"/interface/xml/AperoConvLongFoc.xml" );
			file2.copy(newFile2);
		}
		if (getParamMain()->getParamApero().getLiberCalib().contains(true)) {
			QString newFile(getParamMain()->getDossier() + getParamMain()->getAperoXML().section(".",-0,-2)+QString("3.xml"));
			deleteFile(newFile);
			QFile file( getParamMain()->getMicmacDir()+"/interface/xml/AperoLiberCalib.xml");
			file.copy(newFile);
		}
		cout << tr("Default parameters loaded.").toStdString() << endl;

		//écriture de l'image maîtresse dans un fichier xml
		int i = getParamMain()->findImg(getParamMain()->getParamApero().getImgMaitresse(),1);
		QString calibFile = getParamMain()->getCorrespImgCalib().at(i).getCalibration();
		if (!FichierMaitresse::ecrire(getParamMain()->getDossier() + getParamMain()->getMaitresseXML(), getParamMain()->getParamApero().getImgMaitresse(), calibFile, getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==4)) {
			getParamMain()->setAvancement(-1);
			return;
		}
		if (getParamMain()->getParamApero().getAutoCalib().count()>0) {
			for (int i=0; i<imagesTriees.count(); i++) {
				if (!FichierMaitresse::ecrire(getParamMain()->getDossier() + getParamMain()->getMaitresseXML().section(".",0,-2)+QString("0_%1.xml").arg(numFocales.at(i)), imagesTriees.at(i).first.at(0), imagesTriees.at(i).second, false)) {
					getParamMain()->setAvancement(-1);
					return;
				}
			}
		}
		cout << tr("Master image saved.").toStdString() << endl;

		//liste des fichiers de calibration interne (on supprime les calibrations non associées à une image, orientable ou non)
		getParamMain()->modifParamPastis().modifCalibFiles().clear();
		for (int i=0; i<getParamMain()->getCorrespImgCalib().count(); i++) {
			QString calibFile = getParamMain()->getCorrespImgCalib().at(i).getCalibration();
			bool contains = false;
			if (getParamMain()->getParamPastis().getCalibFiles().count()!=0) {
				for (int j=0; j<getParamMain()->getParamPastis().getCalibFiles().count(); j++) {
					if (getParamMain()->getParamPastis().getCalibFiles().at(j).first==calibFile) {
						contains = true;
						break;
					}
				}
			}
			if (contains) continue;
			bool ok = false;
			int focale = calibFile.section(".",0,-2).right(3).toInt(&ok);
			if (!ok) {
				getParamMain()->setAvancement(-5);
				return;
			}
			getParamMain()->modifParamPastis().modifCalibFiles().push_back(pair<QString, int>(calibFile,focale));
		}		

		//liste des images à orienter
		if (getParamMain()->getParamApero().getAutoCalib().count()>0) {
			//auto-calibration
			for (int i=0; i<imagesTriees.count(); i++) {
				if (!FichierImgToOri::ecrire(getParamMain()->getDossier()+getParamMain()->getImgOriAutoCalibXML().section(".",0,-2)+QString("%1.xml").arg(numFocales.at(i)), imagesTriees.at(i).first, imagesTriees.at(i).first.at(0), imagesTriees.at(i).second, numFocales.at(i))) {
					setReaderror(conv(tr("for autocalibration")));
					getParamMain()->setAvancement(-2);
					return;
				}
			}
		}
		bool withGPSSummit = (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==4);
		if (!getParamMain()->getParamApero().getMultiechelle()) {
			//orientation mono-échelle
			if (!FichierImgToOri::ecrire(getParamMain()->getDossier()+getParamMain()->getImgOriXML(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgMaitresse(), getParamMain()->getParamPastis().getCalibFiles(), withGPSSummit)) {
				setReaderror(QString());
				getParamMain()->setAvancement(-2);
				return;
			}
		} else {
			//orientation multi-échelle
				//courtes focales
			if (!FichierImgToOri::ecrire(getParamMain()->getDossier()+getParamMain()->getImgsCourtOriXML(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgMaitresse(), getParamMain()->getParamPastis().getCalibFiles(), withGPSSummit, false, 0, getParamMain()->getParamApero().getCalibFigees())) {
				setReaderror(conv(tr("at the first step of multiscale computing")));
				getParamMain()->setAvancement(-2);
				return;
			}
				//longues focales (+ courtes avec valeur init)
			if (!FichierImgToOri::ecrire(getParamMain()->getDossier()+getParamMain()->getImgsOriVInitXML(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgMaitresse(), getParamMain()->getParamPastis().getCalibFiles(), withGPSSummit, false, 1, getParamMain()->getParamApero().getCalibFigees())) {
				setReaderror(conv(tr("at the second step of multiscale computing")));
				getParamMain()->setAvancement(-2);
				return;
			}
				//courtes focales figées (2nde partie)
			if (!FichierPosesFigees::ecrire(getParamMain()->getDossier()+getParamMain()->getPosesFigeesXML(), getParamMain()->getParamApero().getCalibFigees(), getParamMain()->getParamPastis().getCalibFiles(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getParamApero().getImgMaitresse(), true)) {
				setReaderror(conv(tr("fixed")));
				getParamMain()->setAvancement(-16);
				return;
			}
				//longues focales libérées (2nde partie)
			if (!FichierPosesFigees::ecrire(getParamMain()->getDossier()+getParamMain()->getPosesLibresXML(), getParamMain()->getParamApero().getCalibFigees(), getParamMain()->getParamPastis().getCalibFiles(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getParamApero().getImgMaitresse(), false)) {
				setReaderror(conv(tr("free")));
				getParamMain()->setAvancement(-16);
				return;
			}
		}
		if (getParamMain()->getParamApero().getLiberCalib().contains(true)) {
			//orientation avec dissociation des calibrations
			if (!FichierImgToOri::ecrire(getParamMain()->getDossier()+getParamMain()->getImgsOriTtInitXML(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgMaitresse(), getParamMain()->getParamPastis().getCalibFiles(), withGPSSummit, false, 2)) {
				setReaderror(conv(tr("with dissociated calibrations")));
				getParamMain()->setAvancement(-2);
				return;
			}
				//calibrations à ne pas dissocier
			if (!FichierPosesFigees::ecrire(getParamMain()->getDossier()+getParamMain()->getPosesNonDissocXML(), getParamMain()->getParamApero().getLiberCalib(), getParamMain()->getParamPastis().getCalibFiles(), getParamMain()->getCorrespImgCalib(), getParamMain()->getParamApero().getImgToOri(), getParamMain()->getParamApero().getImgMaitresse())) {
				setReaderror(conv(tr("with dissociated calibrations")));
				getParamMain()->setAvancement(-16);
				return;
			}
		}
		cout << tr("Images to be oriented saved.").toStdString() << endl;

		//lecture des calibrations non créées pour avoir le type
		for (int i=0; i<getParamMain()->getParamPastis().getCalibFiles().count(); i++) {
			if (getParamMain()->getParamPastis().getCalibs().contains(getParamMain()->getParamPastis().getCalibFiles().at(i).first)) continue;
			CalibCam c;
			QString err = FichierCalibCam::lire(getParamMain()->getDossier(), getParamMain()->getParamPastis().getCalibFiles().at(i).first, c, getParamMain()->getParamPastis().getCalibFiles().at(i).second);
			if (!err.isEmpty()) {
				setReaderror(getParamMain()->getParamPastis().getCalibFiles().at(i).first);
				getParamMain()->setAvancement(-25);
				return;				
			}
			getParamMain()->modifParamPastis().modifCalibs().push_back(c);
		}

		//définition des calibrations internes
			//séparation des calib classiques et fish-eye
			QList<int> lc, lfe;
			for (int i=0; i<getParamMain()->getParamPastis().getCalibs().count(); i++) {
				if (getParamMain()->getParamPastis().getCalibs().at(i).getType()==0)
					lc.push_back(getParamMain()->getParamPastis().getCalibs().at(i).getFocale());
				else
					lfe.push_back(getParamMain()->getParamPastis().getCalibs().at(i).getFocale());
			}
			//définition des calibrations (y compris dans le cas multi-échelle => pour les contraintes)
				//calib classiques
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibClassiqXML(), lc)) {
				setReaderror(conv(tr("conical")));
				getParamMain()->setAvancement(-17);
				return;
			}
				//calib fish-eye
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibFishEyeXML(), lfe)) {
				setReaderror(conv(tr("fish-eye")));
				getParamMain()->setAvancement(-17);
				return;
			}
			//cas mono-échelle (y compris auto-calibration)			
		if (!getParamMain()->getParamApero().getMultiechelle() || getParamMain()->getParamApero().getAutoCalib().count()>0) {
				//définitions de toutes les calibrations
			if (!FichierDefCalibration::ecrire(getParamMain()->getDossier()+getParamMain()->getCalibDefXML(), getParamMain()->getParamPastis().getCalibFiles())) {
				setReaderror(QString());
				getParamMain()->setAvancement(-3);
				return;
			}
		}
			//cas multi-échelle			
		 if (getParamMain()->getParamApero().getMultiechelle()) {
				//définitions de toutes les calibrations courtes focales
			if (!FichierDefCalibration::ecrire(getParamMain()->getDossier()+getParamMain()->getDefCalibCourtXML(), getParamMain()->getParamPastis().getCalibFiles(), false, 0, getParamMain()->getParamApero().getCalibFigees())) {
				setReaderror(conv(tr("for the first step of multiscale computing")));
				getParamMain()->setAvancement(-3);
				return;
			}
				//définitions de toutes les calibrations longues focales
			if (!FichierDefCalibration::ecrire(getParamMain()->getDossier()+getParamMain()->getDefCalibVInitXML(), getParamMain()->getParamPastis().getCalibFiles(), false, 1, getParamMain()->getParamApero().getCalibFigees())) {
				setReaderror(conv(tr("for the second step of multiscale computing")));
				getParamMain()->setAvancement(-3);
				return;
			}
				//clé calib courtes focales
					//toutes confondues
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibCourtXML(), getParamMain()->getParamApero().getCalibFigees())) {
				setReaderror(conv(tr("with short focal length")));
				getParamMain()->setAvancement(-17);
				return;
			}
				QList<int> lcc, lcfe;
				for (int i=0; i<getParamMain()->getParamPastis().getCalibFiles().count(); i++) {
					if (!getParamMain()->getParamApero().getCalibFigees().contains(getParamMain()->getParamPastis().getCalibFiles().at(i).second)) continue;
					if (lc.contains(getParamMain()->getParamPastis().getCalibFiles().at(i).second))
						lcc.push_back(getParamMain()->getParamPastis().getCalibFiles().at(i).second);
					else
						lcfe.push_back(getParamMain()->getParamPastis().getCalibFiles().at(i).second);
				}
					//clé calib courtes focales classiques				
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibCourtClassiqXML(), lcc)) {
				setReaderror(conv(tr("conical with short focal length")));
				getParamMain()->setAvancement(-17);
				return;
			}
					//clé calib courtes focales fish-eye
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibCourtFishEyeXML(), lcfe)) {
				setReaderror(conv(tr("fish-eye with short focal length")));
				getParamMain()->setAvancement(-17);
				return;
			}
				//clé calib longues focales
				QList<int> llc, llfe;
				for (int i=0; i<getParamMain()->getParamPastis().getCalibFiles().count(); i++) {
					if (getParamMain()->getParamApero().getCalibFigees().contains(getParamMain()->getParamPastis().getCalibFiles().at(i).second)) continue;
					if (lc.contains(getParamMain()->getParamPastis().getCalibFiles().at(i).second))
						llc.push_back(getParamMain()->getParamPastis().getCalibFiles().at(i).second);
					else
						llfe.push_back(getParamMain()->getParamPastis().getCalibFiles().at(i).second);
				}
					//clé calib longues focales classiques				
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibLongClassiqXML(), llc)) {
				setReaderror(conv(tr("conical with long focal length")));
				getParamMain()->setAvancement(-17);
				return;
			}
					//clé calib longues focales fish-eye
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibLongFishEyeXML(), llfe)) {
				setReaderror(conv(tr("fish-eye with long focal length")));
				getParamMain()->setAvancement(-17);
				return;
			}
		}
			//cas libération de calibrations
		if (getParamMain()->getParamApero().getLiberCalib().contains(true)) {
				//définitions de toutes les calibrations à libérer
			if (!FichierDefCalibration::ecrire(getParamMain()->getDossier()+getParamMain()->getDefCalibTtInitXML(), getParamMain()->getParamPastis().getCalibFiles(), getParamMain()->getParamApero().getLiberCalib())) {
				setReaderror(conv(tr("dissociated")));
				getParamMain()->setAvancement(-3);
				return;
			}
				//clé calib à libérer
				QList<int> ldc, ldfe;
				for (int i=0; i<getParamMain()->getParamPastis().getCalibFiles().count(); i++) {
					if (!getParamMain()->getParamApero().getLiberCalib().at(i)) continue;
					if (lc.contains(getParamMain()->getParamPastis().getCalibFiles().at(i).second))
						ldc.push_back(getParamMain()->getParamPastis().getCalibFiles().at(i).second);
					else
						ldfe.push_back(getParamMain()->getParamPastis().getCalibFiles().at(i).second);
				}
					//clé calib à libérer classiques			
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibLiberClassiqXML(), ldc)) {
				setReaderror(conv(tr("conical to be dissociated")));
				getParamMain()->setAvancement(-17);
				return;
			}
					//clé calib à libérer fish-eye		
			if (!FichierCleCalib::ecrire(getParamMain()->getDossier()+getParamMain()->getCleCalibLiberFishEyeXML(), ldfe)) {
				setReaderror(conv(tr("fish-eye to free")));
				getParamMain()->setAvancement(-17);
				return;
			}
		}
		cout << tr("Calibration definitions saved.").toStdString() << endl;

		//écriture de l'orientation absolue (si pas d'orientation : fichiers vides)
		QStringList imgsABasculer;
		if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()<3) {				
			if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getDefObsGPSXML())
			|| !createEmptyFile(getParamMain()->getDossier()+getParamMain()->getDefIncGPSXML())
			|| !createEmptyFile(getParamMain()->getDossier()+getParamMain()->getPonderationGPSXML())
			|| !createEmptyFile(getParamMain()->getDossier()+getParamMain()->getOrientationGPSXML())) {
				getParamMain()->setAvancement(-18);
				return;
			}
		} else {	
			//conversion des fichiers de données
			if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==3) {
				QString nextFile = getParamMain()->getDossier()+getParamMain()->getParamApero().getUserOrientation().getPointsGPS().section("/",-1,-1);
				if (nextFile.contains(".")) nextFile = nextFile.section(".",0,-2);
				nextFile = nextFile+QString(".xml");
				if (!FichierAppuiGPS::convert(getParamMain()->getParamApero().getUserOrientation().getPointsGPS(), getParamMain()->getDossier()+QString("tempo.xml"))) {
					setReaderror(getParamMain()->getParamApero().getUserOrientation().getPointsGPS());
					getParamMain()->setAvancement(-22);
					return;
				}
				QFile(nextFile).remove();
				QFile(getParamMain()->getDossier()+QString("tempo.xml")).rename(nextFile);
				getParamMain()->modifParamApero().modifUserOrientation().setPointsGPS(nextFile);
				nextFile = getParamMain()->getDossier()+getParamMain()->getParamApero().getUserOrientation().getAppuisImg().section("/",-1,-1);
				if (nextFile.contains(".")) nextFile = nextFile.section(".",0,-2);
				nextFile = nextFile+QString(".xml");
				if (!FichierAppuiImage::convert(getParamMain()->getParamApero().getUserOrientation().getAppuisImg(), getParamMain()->getDossier()+QString("tempo.xml"))) {
					setReaderror(getParamMain()->getParamApero().getUserOrientation().getPointsGPS());
					getParamMain()->setAvancement(-23);
					return;
				}
				QFile(nextFile).remove();
				QFile(getParamMain()->getDossier()+QString("tempo.xml")).rename(nextFile);
				getParamMain()->modifParamApero().modifUserOrientation().setAppuisImg(nextFile);
			} else {
				QString resultDir;
				if (!FichierSommetsGPS::convert(getParamMain()->getParamApero().getUserOrientation().getPointsGPS(), *getParamMain(), resultDir, imgsABasculer)) {
					setReaderror(getParamMain()->getParamApero().getUserOrientation().getPointsGPS());
					getParamMain()->setAvancement(-22);
					return;
				}
				getParamMain()->modifParamApero().modifUserOrientation().setPointsGPS(resultDir);				
			}

			if (!FichierObsGPS::ecrire(getParamMain()->getDossier()+getParamMain()->getDefObsGPSXML(), getParamMain()->getParamApero().getUserOrientation())) {
				if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==3) setReaderror(conv(tr("(ground-point definition)")));
				else setReaderror(conv(tr("(definition of the camera GPS coordinates)")));
				getParamMain()->setAvancement(-18);
				return;
			}
			if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==3) {
				if(!FichierIncGPS::ecrire(getParamMain()->getDossier()+getParamMain()->getDefIncGPSXML(), getParamMain()->getParamApero().getUserOrientation())) {
					setReaderror(conv(tr("(image measurement definition)")));
					getParamMain()->setAvancement(-18);
					return;
				}
			} else if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getDefIncGPSXML())) {
				getParamMain()->setAvancement(-18);
				return;
			}
			if (!FichierPondGPS::ecrire(getParamMain()->getDossier()+getParamMain()->getPonderationGPSXML(), getParamMain()->getParamApero().getUserOrientation().getOrientMethode())) {
				if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==3) setReaderror(conv(tr("(GCP weight definition)")));
				else setReaderror(conv(tr("(camera GPS coordinate weight definition)")));
				getParamMain()->setAvancement(-18);
				return;
			}
		}
				
		if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getOrientationAbsolueXML())
		|| !createEmptyFile(getParamMain()->getDossier()+getParamMain()->getOrientationGPSXML())) {
			getParamMain()->setAvancement(-18);
			return;
		}
		if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()!=0) {
			QString fichierOriAbs = (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()>=3)? getParamMain()->getOrientationGPSXML() : getParamMain()->getOrientationAbsolueXML();
			if (!FichierBasculOri::ecrire(getParamMain()->getDossier()+fichierOriAbs, getParamMain()->getParamApero().getUserOrientation(), imgsABasculer)) {
				if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==1) setReaderror(conv(tr("(plan, axis, scale)")));
				else if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==2) setReaderror(conv(tr("(image and scale)")));
				else if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==3) setReaderror(conv(tr("(ground-points)")));
				else if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==4) setReaderror(conv(tr("(camera GPS coordinates)")));
				getParamMain()->setAvancement(-18);
				return;
			}
		}
		cout << tr("Georeferencing files saved.").toStdString() << endl;

		//orientation initiale
		if (getParamMain()->getParamApero().getUseOriInit()) {
			if (!FichierOriInit::ecrire(getParamMain()->getDossier()+getParamMain()->getOriInitXML())) {
				getParamMain()->setAvancement(-26);
				return;
			}
		} else {
			if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getOriInitXML())) {
				getParamMain()->setAvancement(-26);
				return;
			}
		}
	
		//export ply (vide au départ, on ne l'exporte qu'une fois, pour la dernière passe)
		if (!getParamMain()->getParamApero().getExportPts3D()) {
			if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getExportPlyXML())) {
				setReaderror(getParamMain()->getExportPlyXML());
				getParamMain()->setAvancement(-24);
				return;
			}
		}
		cout << tr("All parameter files saved.").toStdString() << endl;

		//progress dialog
		getParamMain()->setAvancement(Filtrage);
		if (*getAnnulation()) {
			return;
		}
	}
	
	if (getParamMain()->getAvancement()==Filtrage) {
		setProgressLabel(tr("Tie point filtering"));

		//filtrage : 5 cas : doFiltrage mais déjà fait => on passe
				//doFiltrage et non fait => test_ISA0
				//doFiltrage et fait mais supprimé => renomination des fichiers
				//!doFiltrage mais déjà fait => renomination des fichiers
				//!doFiltrage et non fait (ou supprimé) => on passe
		//vérification des données (et renomination)
		int filtrageDone = 0;	//0 : non fait, 1 : supprimé, 2 : fait
		QString str(getParamMain()->getDossier()+QString("Homol/"));
		QStringList l0 = QDir(str).entryList(QDir::Dirs);
		QStringList l;
		for (int i=0; i<l0.count(); i++) {
			QString str2(str+l0.at(i)+QString("/"));
			QStringList l1 = QDir(str2).entryList(QStringList("*.dat"), QDir::Files);
			for (int j=0; j<l1.count(); j++)
				l.push_back(str2+l1.at(j));
		}
		for (int i=0; i<2; i++) {
			QString str2(l.at(i));
			if (str2.right(9)==QString("_init.dat")) {
				filtrageDone = 2;
				break;
			} else if (str2.right(11)==QString("_filtre.dat")) {
				filtrageDone = 1;
				break;
			}
		}

		//traitement			
		if (getParamMain()->getParamApero().getFiltrage() && filtrageDone==1) {	//cas 3
			for (int i=0; i<l.count(); i++) {	//x -> _init
				QString str2(l.at(i));
				if (str2.right(11)!=QString("_filtre.dat")) {
					QString str3 = str2.section(".",0,-2)+QString("_init.dat");
					QFile(str2).rename(str3);
				}
			}
			for (int i=0; i<l.count(); i++) {	//_filtre -> x
				QString str2(l.at(i));
				if (str2.right(11)==QString("_filtre.dat")) {
					QString str3 = str2.section("_filtre",0,-2)+QString(".dat");
					QFile(str2).rename(str3);
				}
			}
		} else if (!getParamMain()->getParamApero().getFiltrage() && filtrageDone==2) {		//cas 4
			for (int i=0; i<l.count(); i++) {	//x -> _filtre
				QString str2(l.at(i));
				if (str2.right(9)!=QString("_init.dat")) {
					QString str3 = str2.section(".",0,-2)+QString("_filtre.dat");
					QFile(str2).rename(str3);
				}
			}
			for (int i=0; i<l.count(); i++) {	//_init -> x
				QString str2(l.at(i));
				if (str2.right(9)==QString("_init.dat")) {
					QString str3 = str2.section("_init",0,-2)+QString(".dat");
					QFile(str2).rename(str3);
				}
			}
		} else if (getParamMain()->getParamApero().getFiltrage() && filtrageDone==0) {		//cas 2
			//écriture du fichier de paramètres
			int idx = getParamMain()->findImg(getParamMain()->getParamApero().getImgMaitresse(),1);
			if (!FichierFiltrage::ecrire(QString("Filtrage.xml"), getParamMain()->getDossier(), getParamMain()->getCorrespImgCalib().at(idx).getTaille())) {	//prendre la taille max/resol
				if (!*getAnnulation()) getParamMain()->setAvancement(-12);
				return;
			}
			cout << tr("Filtering parameters saved.").toStdString() << endl;
			//filtrage
			QString commande =  comm(QString("cd %1 && %1bin/test_ISA0 %2%3").arg(noBlank(getMicmacDir())).arg(noBlank(getParamMain()->getDossier())).arg(QString("Filtrage.xml")));
			if (execute(commande)!=0) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-13);
				return;
			}
			//renomination des résultats
			FichierFiltrage::renameFiles(getParamMain()->getDossier());
			getParamMain()->modifParamApero().setFiltrage(false);
			cout << tr("Tie points filtered.").toStdString() << endl;
		} else
			cout << tr("No filtering needed").toStdString() << endl;

		//progress dialog
		if (getParamMain()->getParamApero().getAutoCalib().count()>0) getParamMain()->setAvancement(AutoCalibration);
		else getParamMain()->setAvancement(Poses);
		if (*getAnnulation()) {
			return;
		}
	}

	if (getParamMain()->getAvancement()==AutoCalibration) {	
		//auto-calibration
		setProgressLabel(tr("Autocalibration"));

		//tri des images par objectif
		QList<pair<QStringList,QString> > imagesTriees;
		QVector<int> numFocales(0);
		trieImgAutoCalib(imagesTriees, numFocales);

		for (int i=0; i<imagesTriees.count(); i++) {
			//réinitialisation des calibrations s'il y a eu auto-calibration
			QString oldfile = imagesTriees.at(i).second.section(".",0,-2)+QString("_init.xml");
			QString newfile = imagesTriees.at(i).second;
			if (QFile(oldfile).exists()) {
				deleteFile(newfile);
				QFile(oldfile).rename(newfile);
			}

			//renomination des fichiers xml utiles
			deleteFile(getParamMain()->getDossier() + getParamMain()->getMaitresseXML().section(".",0,-2)+QString("0.xml"));
			deleteFile(getParamMain()->getDossier()+getParamMain()->getImgOriAutoCalibXML());
			QFile(getParamMain()->getDossier() + getParamMain()->getMaitresseXML().section(".",0,-2)+QString("0_%1.xml").arg(numFocales.at(i))).copy(getParamMain()->getDossier() + getParamMain()->getMaitresseXML().section(".",0,-2)+QString("0.xml"));
			QFile(getParamMain()->getDossier()+getParamMain()->getImgOriAutoCalibXML().section(".",0,-2)+QString("%1.xml").arg(numFocales.at(i))).copy(getParamMain()->getDossier()+getParamMain()->getImgOriAutoCalibXML());

			//calcul
			#if defined Q_WS_WIN
				if (!QDir(getParamMain()->getDossier()+QString("Ori-F-2")).exists())
					QDir(getParamMain()->getDossier()).mkdir("Ori-F-2");
			#endif
			QString commande = comm(noBlank(getMicmacDir()) + QString("bin/Apero ") + noBlank(getParamMain()->getDossier()) + getParamMain()->getAperoXML().section(".",0,-2)+QString("0.xml") + QString(" >")+noBlank(getStdoutfile()));
			if (execute(commande)!=0) {
				if (!*getAnnulation()) {	
					setReaderror(conv(tr("for autocalibration (focal length %1)")).arg(numFocales.at(i)));
					getParamMain()->setAvancement(-4);
				}
				return;
			}
			//vérification
			QString err = checkAperoOutStream();
			if (!err.isEmpty()) {
				if (!*getAnnulation()) {	
					setReaderror(err);
					getParamMain()->setAvancement(-21);
				}
				return;
			}
			//on renomme le dossier
			deleteFile(getParamMain()->getDossier()+QString("Ori-AutoCalib%1").arg(numFocales.at(i)), true);
			bool b = QFile(getParamMain()->getDossier()+QString("Ori-F-2")).rename(getParamMain()->getDossier()+QString("Ori-AutoCalib%1").arg(numFocales.at(i)));
			if (!b) {
				if (!*getAnnulation()) {	
					setReaderror(QString("Ori-F-2"));
					getParamMain()->setAvancement(-15);
				}
				return;
			}
			//on renomme les calibrations initiales
			deleteFile(getParamMain()->getDossier()+imagesTriees.at(i).second.section(".",0,-2)+QString("_init.xml"));
			QFile(getParamMain()->getDossier()+imagesTriees.at(i).second).rename(getParamMain()->getDossier()+imagesTriees.at(i).second.section(".",0,-2)+QString("_init.xml"));
			//on recopie les nouvelles calibrations
			QString sfocale = QVariant(numFocales.at(i)).toString();
			while (sfocale.count()<3)
				sfocale = QString("0")+sfocale;
			deleteFile(getParamMain()->getDossier()+imagesTriees.at(i).second);
			QFile(getParamMain()->getDossier()+QString("Ori-AutoCalib%1/F%2_AutoCalFinale2.xml").arg(numFocales.at(i)).arg(sfocale)).copy(getParamMain()->getDossier()+imagesTriees.at(i).second);
			cout << tr("Autocalibration (focal length %1) computed.").arg(numFocales.at(i)).toStdString() << endl;
		}
		cout << tr("Auto-calibration computed.").toStdString() << endl;
		getParamMain()->setAvancement(Poses);
	}

	if (getParamMain()->getAvancement()==Poses) {	
		//calcul
		if (!getParamMain()->getParamApero().getMultiechelle()) {
			setProgressLabel(tr("Orientation computing"));

			if (getParamMain()->getParamApero().getExportPts3D() && !getParamMain()->getParamApero().getLiberCalib().contains(true)) {				
				if (!FichierExportPly::ecrire(getParamMain()->getDossier()+getParamMain()->getExportPlyXML())) {
					setReaderror(getParamMain()->getExportPlyXML());
					getParamMain()->setAvancement(-24);
					return;
				}
			}

			#if defined Q_WS_WIN
				if (!QDir(getParamMain()->getDossier()+QString("Ori-F")).exists())
					QDir(getParamMain()->getDossier()).mkdir("Ori-F");
			#endif
			QString commande = comm(noBlank(getMicmacDir()) + QString("bin/Apero ") + noBlank(getParamMain()->getDossier()) + getParamMain()->getAperoXML() + QString(" >")+noBlank(getStdoutfile()));	
			if (execute(commande)!=0) {
				if (!*getAnnulation()) {	
					setReaderror(QString());
					getParamMain()->setAvancement(-4);
				}
				return;
			}	
			//vérification
			QString err = checkAperoOutStream();
			if (!err.isEmpty()) {
				if (!*getAnnulation()) {	
					setReaderror(err);
					getParamMain()->setAvancement(-21);
				}
				return;
			}
			cout << tr("Poses computed").toStdString() << endl;

			//maj interface
			if (getParamMain()->getParamApero().getLiberCalib().contains(true))
				getParamMain()->setAvancement(DissocCalib);
			else
				getParamMain()->setAvancement(ParamVue3D);
		} else {
			setProgressLabel(tr("Orientation computing\nShort focal lengths"));

			#if defined Q_WS_WIN
				if (!QDir(getParamMain()->getDossier()+QString("Orient")).exists())
					QDir(getParamMain()->getDossier()).mkdir("Orient");
			#endif
			QString commande = comm(noBlank(getMicmacDir()) + QString("bin/Apero ") + noBlank(getParamMain()->getDossier()) + getParamMain()->getAperoXML() + QString(" >")+noBlank(getStdoutfile()));
			if (execute(commande)!=0) {
				 if (!*getAnnulation()) {	
					setReaderror(conv(tr("with short focal length")));
					getParamMain()->setAvancement(-4);
				}
				return;
			}
			//vérification
			QString err = checkAperoOutStream();
			if (!err.isEmpty()) {
				if (!*getAnnulation()) {	
					setReaderror(err);
					getParamMain()->setAvancement(-21);
				}
				return;
			}
			cout << tr("Poses with short focal length computed").toStdString() << endl;

			//maj interface
			getParamMain()->setAvancement(PosesLgFocales);
		}
	}

	if (getParamMain()->getAvancement()==PosesLgFocales) {	
		//calcul
		setProgressLabel(tr("Orientation computing\nLong focal lengths"));

		if (getParamMain()->getParamApero().getExportPts3D() && !getParamMain()->getParamApero().getLiberCalib().contains(true)) {				
			if (!FichierExportPly::ecrire(getParamMain()->getDossier()+getParamMain()->getExportPlyXML())) {
				setReaderror(getParamMain()->getExportPlyXML());
				getParamMain()->setAvancement(-24);
				return;
			}
		}

		#if defined Q_WS_WIN
			if (!QDir(getParamMain()->getDossier()+QString("Ori-F")).exists())
				QDir(getParamMain()->getDossier()).mkdir("Ori-F");
		#endif
		QString commande = comm(noBlank(getMicmacDir()) + QString("bin/Apero ") + noBlank(getParamMain()->getDossier()) + getParamMain()->getAperoXML().section(".",0,-2)+QString("2.xml") + QString(" >")+noBlank(getStdoutfile()));
		if (execute(commande)!=0) {
			if (!*getAnnulation()) {	
				setReaderror(conv(tr("with long focal lengths")));
				getParamMain()->setAvancement(-4);
			}
			return;
		}
		//vérification
		QString err = checkAperoOutStream();
		if (!err.isEmpty()) {
			if (!*getAnnulation()) {	
				setReaderror(err);
				getParamMain()->setAvancement(-21);
			}
			return;
		}
		cout << tr("Poses with long focal length computed").toStdString() << endl;

		//maj interface
		if (getParamMain()->getParamApero().getLiberCalib().contains(true))
			getParamMain()->setAvancement(DissocCalib);
		else
			getParamMain()->setAvancement(ParamVue3D);
	}

	if (getParamMain()->getAvancement()==DissocCalib) {
		if (getParamMain()->getParamApero().getLiberCalib().contains(true)) {
			//calcul
			setProgressLabel(tr("Orientation computing\nCalibration dissociation"));
			
			if (getParamMain()->getParamApero().getExportPts3D()) {
				if (!FichierExportPly::ecrire(getParamMain()->getDossier()+getParamMain()->getExportPlyXML())) {
					setReaderror(getParamMain()->getExportPlyXML());
					getParamMain()->setAvancement(-24);
					return;
				}
			}

			#if defined Q_WS_WIN
				if (!QDir(getParamMain()->getDossier()+QString("Ori-F-3")).exists())
					QDir(getParamMain()->getDossier()).mkdir("Ori-F-3");
			#endif
			QString commande = comm(noBlank(getMicmacDir()) + QString("bin/Apero ") + noBlank(getParamMain()->getDossier()) + getParamMain()->getAperoXML().section(".",0,-2)+QString("3.xml") + QString(" >")+noBlank(getStdoutfile()));
			if (execute(commande)!=0) {
				if (!*getAnnulation()) {	
					setReaderror(conv(tr("with dissociated calibrations")));
					getParamMain()->setAvancement(-4);
				}
				return;
			}
			//vérification
			QString err = checkAperoOutStream();
			if (!err.isEmpty()) {
				if (!*getAnnulation()) {	
					setReaderror(err);
					getParamMain()->setAvancement(-21);
				}
				return;
			}
			//renomination du résultat		
			if (QDir(getParamMain()->getDossier()+QString("Ori-F/")).exists()) {
				deleteFile(getParamMain()->getDossier()+QString("Ori-Interm"), true);
				bool b = QFile(getParamMain()->getDossier()+QString("Ori-F")).rename(getParamMain()->getDossier()+QString("Ori-Interm"));
				if (!b) {
					setReaderror("Ori-F");
					getParamMain()->setAvancement(-15);
					return;
				}
			}	
			if (QDir(getParamMain()->getDossier()+QString("Ori-F-3/")).exists()) {
				deleteFile(getParamMain()->getDossier()+QString("Ori-F"), true);
				bool b = QFile(getParamMain()->getDossier()+QString("Ori-F-3")).rename(getParamMain()->getDossier()+QString("Ori-F"));
				if (!b) {
					setReaderror("Ori-F-3");
					getParamMain()->setAvancement(-15);
					return;
				}
			}
			cout << tr("Poses with dissociated calibrations computed").toStdString() << endl;
		}


		//maj interface
		getParamMain()->setAvancement(ParamVue3D);
	}

	if (getParamMain()->getAvancement()==ParamVue3D) {
		emit saveCalcul();
		//on récupère les paramètres de chaque caméra dans les fichier Orient et les points homologues trouvés et on calcule leurs coordonnées 3D
		if (getParamMain()->getParamApero().getCalcPts3D()) setProgressLabel(tr("3D tie point saving"));
		else setProgressLabel(conv(tr("3D view parameter computing")));
		getParamMain()->setEtape(0);

		//lecture des caméras
		QVector<Pose> cameras(getParamMain()->getParamApero().getImgToOri().count());
	        QString err = VueChantier::convert (getParamMain(), cameras, -1);
		if (!err.isEmpty()) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-6);
			setReaderror(err);
			return;
		}
			//cas de l'orientation absolue à partir du géoréférencement d'une image : il faut changer "manuellement" les fichiers Ori
		err = orientationAbsolue();
		if (!err.isEmpty()) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-6);
			setReaderror(err);
			return;
		}
		if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==2) {	//il faut recharger les caméras
			cameras.clear();
			cameras.resize(getParamMain()->getParamApero().getImgToOri().count());
		        err = VueChantier::convert (getParamMain(), cameras, -1);
			if (!err.isEmpty()) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-6);
				setReaderror(err);
				return;
			}
		}

		cout << tr("Cameras read.").toStdString() << endl;
		getParamMain()->setEtape(1);

		//dossier Orient/3D/
		QString dir(getParamMain()->getDossier() + QString("Ori-F/") + QString("3D/"));
		QDir directory(dir);
		directory.cdUp();
		if (!QDir(dir).exists()) {
			bool b = directory.mkdir(QString("3D"));
			if (!b) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-7);
				return;
			}
			cout << tr("Directory 3D created.").toStdString() << endl;
		}

		//suite
			//initialisation
		echelle = numeric_limits<double>::max();
		for (int i=0; i<6; i=i+2) {
			zoneChantier[i] = numeric_limits<double>::max();	//zoneChantier : xmin, xmax, ymin, ymax, zmin, zmax
			zoneChantier[i+1] = -numeric_limits<double>::max();
			zoneChantierEtCam[i] = numeric_limits<double>::max();
			zoneChantierEtCam[i+1] = -numeric_limits<double>::max();
		}

		//objectifs fish-eye
		QVector<bool> fisheye(cameras.count(),false);
		for (int i=0; i<cameras.count(); i++) {
			if (cameras.at(i).getCamera().ExportCalibInterne2XmlStruct(Pt2di(cameras.at(i).width(),cameras.at(i).height())).RayonUtile().IsInit()) fisheye[i] = true;
		}

		//extraction des fichiers de points homologues
		cTplValGesInit<string>  aTpl;
		char** argv = new char*[1];
		char c_str[] = "rthsrth";
		argv[0] = new char[strlen( c_str )+1];
		strcpy( argv[0], c_str );
		cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, getParamMain()->getDossier().toStdString(), aTpl );
		const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");
		vector<pair<string,string> > aPair(aVN->size());
		for (int aK=0; aK<signed(aVN->size()) ; aK++) {
		  	aPair[aK] = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);
		}
			
		//lancement de tous les threads	-> calcul des points 3D et des paramètres globaux
		QVector<Points3DThread> points3DThread(cameras.count(), Points3DThread(getParamMain(), cameras, getAnnulation(), getParamMain()->getParamApero(), aVN, &aPair, &fisheye));
		for (int i=0; i<cameras.count(); i++)
		        points3DThread[i].setIdx(i);
		int nbProc = cpu_count;
		for (int N=0; N<cameras.count(); N+=nbProc) {
			for (int i=0; i<min(nbProc,cameras.count()-N); i++) 
				points3DThread[N+i].start();

			for (int i=0; i<min(nbProc,cameras.count()-N); i++)
				while (points3DThread[N+i].isRunning()) {}
		}

		delete [] argv[0];
		delete [] argv;
		delete mICNM;
		aPair.clear();

		/*for (int i=0; i<cameras.count(); i++) {
			if (fisheye.at(i)) continue;
			points3DThread[i].start();
		}
		for (int i=0; i<cameras.count(); i++) {
			while (points3DThread[i].isRunning()) {}
		}

		for (int i=0; i<cameras.count(); i++) {
			if (!fisheye.at(i)) continue;
			points3DThread[i].start();
			while (points3DThread[i].isRunning()) {}
		}*/

		for (int i=0; i<cameras.count(); i++)
			if (!points3DThreadFinished(points3DThread.at(i))) return;
	
		//enregistrement des paramètres globaux du chantier 3D
		QFile file2(dir + QString("param_chantier"));
		if (!file2.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
			if (!*getAnnulation()) getParamMain()->setAvancement(-9);
			return;
		}
		QTextStream outStream2(&file2);
		outStream2 << echelle << "\n";
		for (int i=0; i<6; i++)
			outStream2 << zoneChantier.at(i) << " ";
		outStream2 << "\n";
		for (int i=0; i<6; i++)
			outStream2 << zoneChantierEtCam.at(i) << " ";
		outStream2 << "\n" << "\n";
                for (int i=0; i<cameras.count(); i++) {
			for (int j=0; j<4; j++) {
				outStream2 << points3DThread.at(i).getEmprise().at(j).x << " " << points3DThread.at(i).getEmprise().at(j).y << " " << points3DThread.at(i).getEmprise().at(j).z << "\n";
			}
			outStream2 << "\n";
		}
                file2.close();
	
		//maj interface
		cout << tr("Done").toStdString() << endl;
		getParamMain()->setAvancement(Termine);
		setProgressLabel(conv(tr("Done")));
	}
}

void AperoThread::trieImgAutoCalib(QList<std::pair<QStringList,QString> >& imagesTriees, QVector<int>& numFocales) {
	if (getParamMain()->getParamApero().getAutoCalib().count()==0) return;
	for (QStringList::const_iterator it=getParamMain()->getParamApero().getAutoCalib().begin(); it!=getParamMain()->getParamApero().getAutoCalib().end(); it++) {
		QString calib = getParamMain()->getCorrespImgCalib().at( getParamMain()->findImg(*it,1) ).getCalibration();
		bool b = false;
		if (imagesTriees.count()>0) {
			for (int j=0; j<imagesTriees.count(); j++) {
				if (imagesTriees.at(j).second==calib) {
					imagesTriees[j].first.push_back(*it);
					b = true;
					break;
				}
			}
		}
		if (!b) imagesTriees.push_back( pair<QStringList,QString>( QStringList(*it) , calib) );
	}
	numFocales.resize(imagesTriees.count());
	for (int i=0; i<imagesTriees.count(); i++)
		numFocales[i] = getParamMain()->getParamPastis().findFocale( imagesTriees.at(i).second );
}

QString AperoThread::checkAperoOutStream() {
//vérifie dans le fichier outstream si le calcul a bien convergé (pas de nan)
	QFile file(getStdoutfile());
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return conv(tr("Fail to open file %1.")).arg(getStdoutfile());
	QTextStream inStream(&file);
	bool b = true;
	while (!inStream.atEnd()) {
		QString text = inStream.readLine();
		if (!text.contains("RESIDU LIAISON MOYENS")) continue;
		if (text.contains("nan")) b = false;
		else b = true;
	}
	file.close();
 	if (!b) return conv(tr("Calculation did not converge."));
	return QString();
}

QString AperoThread::orientationAbsolue() {
//cas le l'orientation absolue par une image de géoréférencement connu : on lit le géoréférencement et on modifie les fichiers Ori pour faire le basculement
	if (getParamMain()->getParamApero().getUserOrientation().getOrientMethode()!=2) return QString();

	//système métrique
	QString virgule;
	QString err = systemeNumerique(virgule);
	if (!err.isEmpty()) return err;

	//récupération de l'orientation absolue de l'image 0
	Pt3dr Sabs0;
	ElMatrix<REAL> Rabs0(3,3,0);
	if (!getParamMain()->getParamApero().getUserOrientation().getGeorefFile().isEmpty()) {
		QString fichierAbs0 = getParamMain()->getParamApero().getUserOrientation().getGeorefFile();
			//changement de fichier selon la virgule
			QString f1;
			if (fichierAbs0.right(5)==QString("2.xml")) f1 = fichierAbs0.left(fichierAbs0.count()-5)+QString(".xml");
			if (!QFile(f1).exists()) f1 = QString();
			QString f2 = fichierAbs0.section(".",0,-2)+QString("2.xml");
			if (!QFile(f2).exists()) f2 = QString();
			if (virgule==QString(".") && !f1.isEmpty()) fichierAbs0 = f1;
			else if (virgule==QString(",") && !f2.isEmpty()) fichierAbs0 = f2;
		CamStenope* cameraAbs0 = NS_ParamChantierPhotogram::Cam_Gen_From_File(fichierAbs0.toStdString(), string("OrientationConique"), 0)->CS();
		if (cameraAbs0==0) return conv(tr("Fail to read camera pose %1.")).arg(fichierAbs0);
		Sabs0 = cameraAbs0->PseudoOpticalCenter();
		Rabs0 = cameraAbs0->Orient().Mat();
	} else {
		const QVector<REAL>* centreAbs0 = &(getParamMain()->getParamApero().getUserOrientation().getCentreAbs());
		const QVector<REAL>* rotAbs0 = &(getParamMain()->getParamApero().getUserOrientation().getRotationAbs());
		Sabs0 = Pt3dr(centreAbs0->at(0), centreAbs0->at(1), centreAbs0->at(2));
		for (int i=0; i<9; i++)
			Rabs0(i/3,i%3) = rotAbs0->at(i);
	}

	//récupération de l'orientation relative de l'image 0
	QString imageRef = getParamMain()->getParamApero().getUserOrientation().getImageGeoref();	
	QString fichierRel0 = getParamMain()->getDossier() + QString("Ori-F/OrFinale-%1.xml").arg(imageRef.section(".",0,-2));
		if (virgule==QString(",")) fichierRel0 = fichierRel0.section(".",0,-2)+QString("2.xml");
	CamStenope* cameraRel0 = NS_ParamChantierPhotogram::Cam_Gen_From_File(fichierRel0.toStdString(), string("OrientationConique"), 0)->CS();
	if (cameraRel0==0) return conv(tr("Fail to read camera pose %1.")).arg(fichierRel0);
	Pt3dr Srel0 = cameraRel0->PseudoOpticalCenter();
	ElMatrix<REAL> Rrel0 = cameraRel0->Orient().Mat();

	//différence d'orientation
		//cas trivial
		bool b = true;
		for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			if (Rabs0(i,j)!=Rrel0(i,j)) {
				b = false;
				i = 3;
				break;
			}
		}}
		if (Sabs0==Srel0 && b) return QString();
	ElMatrix<REAL> rotation(3,3,0); 	//rotation = Rabs0 * Rrel0^(⁻1)
		//déterminant
		double determinant = 0;
		for (int i=0; i<3; i++) {
			double d = 1;
			double d2 = 1;
			for (int j=0; j<3; j++) {
				d *= Rrel0((j+i)%3,j);
				d2 *= Rrel0((j-i+3)%3,2-j);
			}
			determinant += d;
			determinant -= d2;
		}
		if (abs(determinant)<pow(10.0,-19)) return conv(tr("Camera %1 determinant is null.")).arg(fichierRel0);
		//inversion
		ElMatrix<REAL> rotInv(3,3);
		for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			double coeff = Rrel0((i+1)%3,(j+1)%3) * Rrel0((i+2)%3,(j+2)%3) - Rrel0((i+1)%3,(j+2)%3) * Rrel0((i+2)%3,(j+1)%3);
			rotInv(j,i) = coeff / determinant;
		}}
	//rotation
	for (int i=0; i<3; i++) {
	for (int j=0; j<3; j++) {
	for (int k=0; k<3; k++) {
		rotation(i,j) += Rabs0(i,k) * rotInv(k,j);
	}}}

	for (int i=0; i<getParamMain()->getParamApero().getImgToOri().count(); i++) {
		//récupération de l'orientation relative de l'image i
		QString fichierRel = getParamMain()->getDossier() + QString("Ori-F/OrFinale-%1.xml").arg(getParamMain()->getParamApero().getImgToOri().at(i).section(".",0,-2));
			if (virgule==QString(",")) fichierRel = fichierRel.section(".",0,-2)+QString("2.xml");
		CamStenope* cameraRel = NS_ParamChantierPhotogram::Cam_Gen_From_File(fichierRel.toStdString(), string("OrientationConique"), 0)->CS();
		if (cameraRel==0) return conv(tr("Fail to read camera pose %1.")).arg(fichierRel);
		Pt3dr Srel = cameraRel->PseudoOpticalCenter();	//à cause des erreurs de précision dans le calcul de rotation, il faut aussi recalculer la position de imgRef (cohérence dans le nuage)
		ElMatrix<REAL> Rrel = cameraRel->Orient().Mat();

		//calcul de l'orientation absolue de l'image i
		QVector<REAL> Sr(0);
		Sr << (Srel.x-Srel0.x) << (Srel.y-Srel0.y) << (Srel.z-Srel0.z);
		QVector<REAL> Sa(3,0);
		for (int j=0; j<3; j++) {
			for (int k=0; k<3; k++)
				Sa[j] += rotation(j,k) * Sr[k];
		}
		Pt3dr Sabs = Pt3dr( Sa[0]+Sabs0.x,  Sa[1]+Sabs0.y,  Sa[2]+Sabs0.z );
		ElMatrix<REAL> Rabs(3,3,0);
		for (int j=0; j<3; j++) {
		for (int k=0; k<3; k++) {
			Rabs(j,k) = 0;
			for (int k=0; k<3; k++)
				Rabs(j,k) += rotation(j,k) * Rrel(k,k);
		}}

		//écriture (fichier *.xml et fichier *2.xml)
		QString fVirg = getParamMain()->getDossier() + QString("Ori-F/OrFinale-%1").arg(getParamMain()->getParamApero().getImgToOri().at(i).section(".",0,-2)) + QString("2.xml");
		QString fPoint = getParamMain()->getDossier() + QString("Ori-F/OrFinale-%1.xml").arg(getParamMain()->getParamApero().getImgToOri().at(i).section(".",0,-2));

		QFile fichierR(fVirg);
		if (!fichierR.open(QIODevice::Text | QIODevice::ReadOnly))
			return conv(tr("Fail to open file %1.")).arg(fVirg);
		QFile fichierRb(applicationPath() + QString("/tempofile.xml"));
		if (!fichierRb.open(QIODevice::Text | QIODevice::WriteOnly | QIODevice::Truncate))
			return conv(tr("fail to write file tempofile.xml."));
		QTextStream instream(&fichierR);
		QTextStream outstream(&fichierRb);
		while (!instream.atEnd()) {
			QString text = instream.readLine();
			if (text.contains("Centre"))
				text = QString("\t\t\t<Centre>%1 %2 %3</Centre>").arg(Sabs.x,0,'g',16).arg(Sabs.y,0,'g',16).arg(Sabs.z,0,'g',16).replace(".",",");
			else for (int j=0; j<3; j++) {
				if (text.contains(QString("L%1").arg(j+1))) {
					text = QString("\t\t\t<L%1>%2 %3 %4</L%1>").arg(j+1).arg(Rabs(j,0),0,'g',16).arg(Rabs(j,1),0,'g',16).arg(Rabs(j,2),0,'g',16).replace(".",",");
					break;	
				}		
			}
			outstream << text << endl;
		}
		fichierR.close();
		fichierRb.close();
		fichierR.remove();
		fichierRb.rename(fVirg);

		QFile fichierR2(fPoint);
		if (!fichierR2.open(QIODevice::Text | QIODevice::ReadOnly))
			return conv(tr("Fail to open file %1.")).arg(fPoint);
		QFile fichierR2b(applicationPath() + QString("/tempofile.xml"));
		if (!fichierR2b.open(QIODevice::Text | QIODevice::WriteOnly | QIODevice::Truncate))
			return conv(tr("fail to write file tempofile.xml."));
		QTextStream instream2(&fichierR2);
		QTextStream outstream2(&fichierR2b);
		while (!instream2.atEnd()) {
			QString text = instream2.readLine();
			if (text.contains("Centre"))
				text = QString("\t\t\t<Centre>%1 %2 %3</Centre>").arg(Sabs.x,0,'g',16).arg(Sabs.y,0,'g',16).arg(Sabs.z,0,'g',16);
			else for (int j=0; j<3; j++) {
				if (text.contains(QString("L%1").arg(j+1))) {
					text = QString("\t\t\t<L%1>%2 %3 %4</L%1>").arg(j+1).arg(Rabs(j,0),0,'g',16).arg(Rabs(j,1),0,'g',16).arg(Rabs(j,2),0,'g',16);
					break;	
				}		
			}
			outstream2 << text << endl;
		}
		fichierR2.close();
		fichierR2b.close();
		fichierR2.remove();
		fichierR2b.rename(fPoint);
	}
	cout << tr("Absolute aerotriangulation is computed.").toStdString() << endl;
	return QString();
}

bool AperoThread::points3DThreadFinished(const Points3DThread& points3DThread) {
	if (!points3DThread.getIsDone()) {
		setReaderror(points3DThread.getReaderror());
		return false;
	}
	echelle = min(echelle, points3DThread.getEchelle());
	for (int j=0; j<6; j=j+2) {
		zoneChantier[j] = min(zoneChantier.at(j), double(points3DThread.getZoneChantier().at(j)));
		zoneChantier[j+1] = max(zoneChantier.at(j+1), double(points3DThread.getZoneChantier().at(j+1)));
		zoneChantierEtCam[j] = min(zoneChantierEtCam.at(j), double(points3DThread.getZoneChantierEtCam().at(j)));
		zoneChantierEtCam[j+1] = max(zoneChantierEtCam.at(j+1), double(points3DThread.getZoneChantierEtCam().at(j+1)));
	}

	getParamMain()->setEtape(getParamMain()->getEtape()+2);
	return true;
}

bool AperoThread::killProcess() {
	bool b = false;
	if (killall("Apero")==0) b = true;
	return b;
}


//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//


Points3DThread::Points3DThread():
	AppliThread( 0, QString(), 0 ),
	cameras( QVector<Pose>( 0 ) ),
	zoneChantier( QVector<GLdouble>( 6 ) ),
	zoneChantierEtCam( QVector<GLdouble>( 6 ) ),
	emprise( QVector<Pt3dr>( 4 ) ),
	paramApero( 0 ),
	aVN( 0 ),
	aPair( 0 ),
	typesCam( 0 ){}

Points3DThread::Points3DThread(ParamMain* pMain, const QVector<Pose>& cam, bool* annul, const ParamApero& pApero, const vector<string>* fichiers, const vector<pair<string,string> >* pairCam, const QVector<bool>* fishEye):
	AppliThread( pMain, QString(), annul ),
	cameras( cam ),
	zoneChantier( QVector<GLdouble>( 6 ) ),
	zoneChantierEtCam( QVector<GLdouble>( 6 ) ),
	emprise( QVector<Pt3dr>( 4 ) ),
	paramApero( &pApero ),
	aVN( fichiers ),
	aPair( pairCam ),
	typesCam( fishEye ){}
	
Points3DThread::Points3DThread(const Points3DThread& points3DThread) : AppliThread() { copie(this, points3DThread); }
Points3DThread::~Points3DThread () {}

Points3DThread& Points3DThread::operator=(const Points3DThread& points3DThread) {
	if (this!=&points3DThread)
		copie(this, points3DThread);
	return *this;
}

void copie(Points3DThread* points3DThread1, const Points3DThread& points3DThread2) {
	copie(dynamic_cast<AppliThread*>(points3DThread1), points3DThread2);
        points3DThread1->cameras = points3DThread2.cameras;
        points3DThread1->idx = points3DThread2.idx;
        points3DThread1->echelle = points3DThread2.echelle;
        points3DThread1->zoneChantier = points3DThread2.zoneChantier;
        points3DThread1->zoneChantierEtCam = points3DThread2.zoneChantierEtCam;
        points3DThread1->emprise = points3DThread2.emprise;
        points3DThread1->paramApero = points3DThread2.paramApero;
        points3DThread1->aVN = points3DThread2.aVN;
        points3DThread1->aPair = points3DThread2.aPair;	
        points3DThread1->typesCam = points3DThread2.typesCam;
}

double Points3DThread::getEchelle() const { return echelle; }
const QVector<GLdouble>& Points3DThread::getZoneChantier() const { return zoneChantier; }
const QVector<GLdouble>& Points3DThread::getZoneChantierEtCam() const { return zoneChantierEtCam; }
const QVector<Pt3dr>& Points3DThread::getEmprise() const { return emprise; }
void Points3DThread::setIdx(int i) { idx=i; }

void Points3DThread::run() {
	setDone(false);
	if (*getAnnulation()) return;

	QList<pair<Pt3dr, QColor> > ptsApp;
	if (paramApero->getCalcPts3D()) {
		//on vérifie que les points n'ont pas déjà été calculés
		QString dir(getParamMain()->getDossier() + QString("Ori-F/") + QString("3D/"));
		QString fichierPoints(dir+cameras.at(idx).getNomImg().section(".",0,-2)+QString(".txt"));
		if (QFile(fichierPoints).exists()) {
			QFile file(fichierPoints);
			if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-8);
				setReaderror(cameras.at(idx).getNomImg());
				return;
			}
			QTextStream inStream(&file);
			while (!inStream.atEnd()) {
				Pt3dr P;
				QVector<int> C(3);
				inStream >> P.x >> P.y >> P.z >> C[0] >> C[1] >> C[2];
				ptsApp.push_back( pair<Pt3dr, QColor>(P, QColor(C[0],C[1],C[2]) ) );
			}
			file.close();
		} else {
			//lecture des points d'appui
				//initialisation chantier
			QImage image(getParamMain()->getDossier()+getParamMain()->convertTifName2Couleur(cameras.at(idx).getNomImg()));
			if (image.isNull()) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-11);
				setReaderror(getParamMain()->getDossier()+getParamMain()->convertTifName2Couleur(cameras.at(idx).getNomImg()));
				return;
			}
				//fichier (*aVN)[aK]
			for (int aK=0; aK<signed(aVN->size()) ; aK++) {	
				if (*getAnnulation()) return;
				//caméras associées
				if (cameras.at(idx).getNomImg()!=QString(aPair->at(aK).first.c_str())) continue;
				int cam2 = -1;
				for (int i=0; i<cameras.count(); i++) {
				        if (cameras.at(i).getNomImg()==QString(aPair->at(aK).second.c_str())) {
						cam2 = i;
						break;
					}
				}
				if (cam2==-1) continue;	//cas notamment des fichiers "init" (sauvegarde avant filtrage)
				//extraction des points d'appui
				ElPackHomologue aPack = ElPackHomologue::FromFile( getParamMain()->getDossier().toStdString()+(*aVN)[aK] );
				if (aPack.size()==0) continue;
				if (*getAnnulation()) return;
				for (ElPackHomologue::const_iterator  itH=aPack.begin(); itH!=aPack.end() ; itH++)
				{				
					Pt2dr pt1(itH->P1());
					Pt2dr pt2(itH->P2());	//itH : coord dans les img rééch, pt = itH(i) * img.width() / 2^(int(log2(sZ)+1))
					QColor c = QColor(image.pixel(  min(max(0,int(pt1.x)),image.width()) , min(max(0,int(pt1.y)),image.height()) ));

					//distorsion
					Pt3dr pt3D(0,0,0);
//					if (!typesCam->at(idx) && !typesCam->at(cam2)) {
//						pt1 = cameras.at(idx).getCamera().F2toC2(pt1);
//						pt2 = cameras.at(cam2).getCamera().F2toC2(pt2);

						//calcul du point 3D
//						double aD;
//						pt3D = cameras.at(idx).getCamera().PseudoInter(pt1, cameras.at(cam2).getCamera(), pt2, &aD);
//					} else {
						std::vector<ElSeg3D> aVS;
						Pt3dr centre1 = cameras.at(idx).centre();
						//Pt3dr centre1(0,0,0);
						Pt3dr centre2 = cameras.at(cam2).centre();
						//Pt3dr centre2(10,0,0);
						Pt3dr aDir1 = cameras.at(idx).getCamera().F2toDirRayonR3(pt1);
						//Pt3dr aDir1(0,0,20);
						Pt3dr aDir2 = cameras.at(cam2).getCamera().F2toDirRayonR3(pt2);
						//Pt3dr aDir2(-10,0,20);
						aVS.push_back(ElSeg3D(centre1,centre1+aDir1));
						aVS.push_back(ElSeg3D(centre2,centre2+aDir2));
						bool ok =true;
						pt3D = ElSeg3D::L2InterFaisceaux(0, aVS, &ok);
						//pt3D = Pt3dr(0,0,20);
						if (!ok || pt3D.x!=pt3D.x || pt3D.y!=pt3D.y || pt3D.z!=pt3D.z) continue;	//teste si coordonnée infinie
//					}

					//erreur
					Pt2dr proj1 = cameras.at(idx).getCamera().R3toF2(pt3D);
					Pt2dr proj2 = cameras.at(cam2).getCamera().R3toF2(pt3D);
					double aD1 = (proj1.x-pt1.x)*(proj1.x-pt1.x) + (proj1.y-pt1.y)*(proj1.y-pt1.y);
					double aD2 = (proj2.x-pt2.x)*(proj2.x-pt2.x) + (proj2.y-pt2.y)*(proj2.y-pt2.y);
					if (aD1>9 || aD2>9) continue;
					//if (aD>3 && getParamMain()->getParamApero().getUserOrientation().getOrientMethode()==0) continue;	//aD dépend de l'échelle (sinon en px) => à modifier dans le cas de la mise à l'échelle
					ptsApp.push_back(pair<Pt3dr, QColor>(pt3D,c));	
				}
			}
			cout << tr("3D points computed").toStdString() << endl;
			if (*getAnnulation()) return;

			//écriture des points 3D
			QFile file(fichierPoints);
			if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
				if (!*getAnnulation()) getParamMain()->setAvancement(-8);
				setReaderror(cameras.at(idx).getNomImg());
				return;
			}
			QTextStream outStream(&file);
			if (ptsApp.count()>0) {
				for (QList<pair<Pt3dr, QColor> >::const_iterator it=ptsApp.begin(); it!=ptsApp.end(); it++)
					outStream << it->first.x << " " << it->first.y << " " << it->first.z << " " << it->second.red() << " " << it->second.green() << " " << it->second.blue() << "\n";
			}
			file.close();	
			cout << tr("3D points saved.").toStdString() << endl;
		}
	}

	//calcul de l'échelle
	echelle = numeric_limits<double>::max();	//min profondeur (= dist min centre - points d'appui) pour le tracer de la caméra
	double x1 = cameras.at(idx).centre().x;
	double y1 = cameras.at(idx).centre().y;
	double z1 = cameras.at(idx).centre().z;
	//distance minimale aux caméras voisines
	double dist1 = numeric_limits<double>::max();
	for (int j=0; j<cameras.count(); j++) {
                double x2 = cameras.at(j).centre().x;
                double y2 = cameras.at(j).centre().y;
                double z2 = cameras.at(j).centre().z;
		double d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
		if (d==0) continue;
		if (d<dist1) dist1 = d;	
	}
	//distance minimale aux points d'appui
	double dist2 = numeric_limits<double>::max();
	if (ptsApp.count()>0) {
		for (QList<pair<Pt3dr, QColor> >::const_iterator it=ptsApp.begin(); it!=ptsApp.end(); it++) {
			double x2 = it->first.x;
			double y2 = it->first.y;
			double z2 = it->first.z;
			double d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
			if (d==0) continue;
			if (d<dist2) dist2 = d;			
		}
	}
	//échelle correspondante
        REAL f = cameras.at(idx).getCamera().Focale();
	double ech = min(sqrt(dist2)/f/0.1, sqrt(dist1)/double(max(cameras.at(idx).width(),cameras.at(idx).height()))*2.0/3.0);
	if (ech<echelle) echelle = ech;
	cout << tr("Survey scale computed.").toStdString() << endl;

	//limites du chantier : 3*écart-type des points homologues (ce qui permet de ne pas compter les points trop loin de la zone d'intérêt)
	GLdouble sigma0[3] = {0,0,0};
	GLdouble moy[3] = {0,0,0};
	GLdouble n =0;
	for (int i=0; i<6; i=i+2) zoneChantier[i] = numeric_limits<double>::max();
	for (int i=1; i<6; i=i+2) zoneChantier[i] = -numeric_limits<double>::max();
		QVector<double> C(3);
		C[0] = cameras.at(idx).centre().x;
		C[1] = cameras.at(idx).centre().y;
		C[2] = cameras.at(idx).centre().z;
	if (ptsApp.count()>0) {
		n += ptsApp.count();
		for (int j=0; j<ptsApp.count(); j++) {
			QVector<double> pt(3);
			pt[0] = ptsApp.at(j).first.x;
			pt[1] = ptsApp.at(j).first.y;
			pt[2] = ptsApp.at(j).first.z;
			for (int k=0; k<3; k++) {
				sigma0[k] += pt[k] * pt[k];
				moy[k] += pt[k];
			}
		}
		GLdouble sigma;
		for (int k=0; k<3; k++) {
			moy[k] /= n;
			sigma0[k] /= n;
			sigma = sqrt(sigma0[k] - moy[k]*moy[k]);
			zoneChantier[2*k] = moy[k] - 3.0*sigma;
			zoneChantier[2*k+1] = moy[k] + 3.0*sigma;
		}
	} else {
		for (int j=0; j<3; j++) {
			if (C[j]<zoneChantier[2*j]) zoneChantier[2*j] = C[j];
			if (C[j]>zoneChantier[2*j+1]) zoneChantier[2*j+1] = C[j];
		}
	}
	for (int k=0; k<6; k++)
		zoneChantierEtCam[k] = zoneChantier.at(k);
	
	//on inclut les caméras
	for (int j=0; j<3; j++) {
		if (C[j]<zoneChantierEtCam[2*j]) zoneChantierEtCam[2*j] = C[j];
		if (C[j]>zoneChantierEtCam[2*j+1]) zoneChantierEtCam[2*j+1] = C[j];
	}
	cout << tr("Survey bouding box computed.").toStdString() << endl;

	//emprises des caméras à la profondeur moyenne des points d'appui
        ElMatrix<REAL> R = cameras.at(idx).rotation();
	//profondeur moyenne
	double z = 0;
	if (ptsApp.count()>0) {
		for (QList<pair<Pt3dr, QColor> >::iterator it=ptsApp.begin(); it!=ptsApp.end(); it++) {
			QVector<double> pt(3);
			pt[0] = it->first.x;
			pt[1] = it->first.y;
			pt[2] = it->first.z;
			for (int j=0; j<3; j++) {
		                z += (pt[j] - C[j]) * R(j,2);
			}
		}
		z /= ptsApp.count();
	} else {
		z = cameras.at(idx).getCamera().GetAltiSol();
	}
	//rectangle
        emprise[0] = cameras.at(idx).getCamera().ImEtProf2Terrain(Pt2dr(0,0),z);
        emprise[1] = cameras.at(idx).getCamera().ImEtProf2Terrain(Pt2dr(cameras.at(idx).width(),0),z);
        emprise[3] = cameras.at(idx).getCamera().ImEtProf2Terrain(Pt2dr(0,cameras.at(idx).height()),z);
        emprise[2] = cameras.at(idx).getCamera().ImEtProf2Terrain(Pt2dr(cameras.at(idx).width(),cameras.at(idx).height()),z);
	cout << tr("Image bounding box computed.").toStdString() << endl;
	setDone(true);
}

//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MicmacThread::MicmacThread (ParamMain* pMain, const QString& stdoutfilename, bool* annul) : AppliThread(pMain,stdoutfilename,annul)
	{ setEndResult(Termine); }

void MicmacThread::run() {
	emit saveCalcul();

	//fichiers de paramètres
	QString geoIXml = getParamMain()->getDossier() + getParamMain()->getMicmacXML();
	QString terXml = getParamMain()->getDossier() + getParamMain()->getMicmacTerXML();
	QString mmorthoXml = getParamMain()->getDossier() + getParamMain()->getMicmacMMOrthoXML();
	deleteFile(geoIXml);
	deleteFile(terXml);
	deleteFile(mmorthoXml);

	for (int i=0; i<getParamMain()->getParamMicmac().count(); i++) {
		//progress dialog
		if (*getAnnulation()) return;
		const CarteDeProfondeur* currentCarte = &(getParamMain()->getParamMicmac().at(i));
		if (!currentCarte->getACalculer()) continue;	//on ne la recalcule pas
		setProgressLabel(tr("Depth map computing\n")+currentCarte->getImageDeReference());

			//paramètres
		bool repere = currentCarte->getRepere();
		bool ok;
		QString numCarte = getParamMain()->getNumImage(currentCarte->getImageDeReference(),&ok,false);
		if (!ok) {
			setReaderror(currentCarte->getImageDeReference());
			getParamMain()->setAvancement(-6);
			return;
		}

		if (getParamMain()->getAvancement()<=Enregistrement) {	
			cout << tr("Depth map %1 parameter saving :\n").arg(currentCarte->getImageDeReference()).toStdString();

			//recopie du fichier des paramètres
			if (repere && !QFile(geoIXml).exists()) QFile(getParamMain()->getMicmacDir()+"/interface/xml/MicMacConver.xml").copy(geoIXml);
			else if (!repere && !QFile(terXml).exists()) QFile(getParamMain()->getMicmacDir()+"/interface/xml/MicMacEuclidien.xml").copy(terXml);
			if (!repere && currentCarte->getDoOrtho() && !QFile(mmorthoXml).exists()) QFile(getParamMain()->getMicmacDir()+"/interface/xml/MicMacOrtho.xml").copy(mmorthoXml);

			//intervalle
			if (!FichierIntervalle::ecrire(getParamMain()->getDossier()+getParamMain()->getIntervalleXML(), currentCarte->getInterv().first, currentCarte->getInterv().second)) {
				getParamMain()->setAvancement(-12);
				return;
			}
			//gestion des discontinuités et des fortes pentes
			if (currentCarte->getDiscontinuites()) {
				if (!FichierDiscontinuites::ecrire(getParamMain()->getDossier()+getParamMain()->getDiscontinuteXML(), currentCarte->getSeuilZ(), currentCarte->getSeuilZRelatif())) {
					getParamMain()->setAvancement(-13);
					return;
				} 
			} else {
				if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getDiscontinuteXML())) {
					getParamMain()->setAvancement(-13);
					return;
				} 
			}

			//effacement des résultats précédents
			QString oldDir = (!repere)? QString("GeoTer%1").arg(numCarte) : QString("GeoI%1").arg(numCarte);
			if (QDir(getParamMain()->getDossier()+oldDir).exists()) {
				bool b = rm(getParamMain()->getDossier() + oldDir);
				if (!b) {
					setReaderror(getParamMain()->getDossier() + oldDir);
					getParamMain()->setAvancement(-10);
					return;
				}
			}
			if (!repere && currentCarte->getDoOrtho()) {
				QString oldDir2 = QString("ORTHO%1").arg(numCarte);
				if (QDir(getParamMain()->getDossier()+oldDir2).exists()) {
					bool b = rm(getParamMain()->getDossier() + oldDir2);
					if (!b) {
						setReaderror(getParamMain()->getDossier() + oldDir2);
						getParamMain()->setAvancement(-10);
						return;
					}
				}
			}
		
			//fichier de définition du masque
			if (!FichierDefMasque::ecrire(getParamMain()->getDossier(), getParamMain()->getDefMasqueXML(), currentCarte->getMasque(*getParamMain()).section("/",-1,-1), currentCarte->getReferencementMasque(*getParamMain()).section("/",-1,-1))) {
				getParamMain()->setAvancement(-2);
				return;
			}
			cout << tr("Mask definition file saved.").toStdString() << endl;

			//liste des images pour la corrélation
			if (!FichierCartes::ecrire(getParamMain()->getDossier()+getParamMain()->getCartesXML(), *currentCarte)) {
				getParamMain()->setAvancement(-14);
				return;
			}
			cout << tr("Fail to save search interval for correlation.").toStdString() << endl;

			//repère
			if (!repere) {
				deleteFile(getParamMain()->getDossier()+getParamMain()->getRepereXML());
				QFile(getParamMain()->getDossier()+currentCarte->getRepereFile(*getParamMain())).copy(getParamMain()->getDossier()+getParamMain()->getRepereXML());
				cout << tr("DTM frame copied.").toStdString() << endl;
			}

			//orthoimages
			if (!repere && currentCarte->getDoOrtho()) {
				if (!FichierOrtho::ecrire(getParamMain()->getDossier()+getParamMain()->getOrthoXML(), *currentCarte, *getParamMain())) {
					getParamMain()->setAvancement(-11);
					return;
				}
				cout << tr("Orthorectification parameters saved.").toStdString() << endl;
			} else if (!repere) {	//fichiers vide
				if (!createEmptyFile(getParamMain()->getDossier()+getParamMain()->getOrthoXML())) {
					getParamMain()->setAvancement(-11);
					return;
				}
			}		

			//fichiers de sortie
			if (!FichierNomCarte::ecrire(getParamMain()->getDossier() + getParamMain()->getNomCartesXML(), numCarte, false, repere)) {
				getParamMain()->setAvancement(-5);
				return;
			}	
			cout << tr("Output parameter file saved.").toStdString() << endl;	
			getParamMain()->setAvancement(Calcul);
		}

		if (getParamMain()->getAvancement()==Calcul) {	
			//calcul cartes
			cout << tr("Depth map %1 computing :\n").arg(currentCarte->getImageDeReference()).toStdString();
			QString paramFile(repere? geoIXml : terXml);
			QString newTempoDir = QString("cd ") + noBlank(getParamMain()->getDossier()) + QString(" & ");
			QString commande = comm(newTempoDir + noBlank(getMicmacDir()) + QString("bin/MICMAC ") + noBlank(paramFile) + QString(" >")+noBlank(getStdoutfile()));
			if (execute(commande)!=0) {
				getParamMain()->setAvancement(-4);
				return;
			}	
			cout << tr("Depth map computed.").toStdString() << endl;	

			//répertoire conversion
			QString oldDir = (!repere)? QString("GeoTer%1").arg(numCarte) : QString("GeoI%1").arg(numCarte);
			QDir directory(getParamMain()->getDossier() + oldDir);
			bool b = directory.mkdir(QString("Conversion"));
			if (!b) {
				setReaderror(oldDir+QString("/Conversion"));
				getParamMain()->setAvancement(-7);
				return;
			}
			cout << tr("Directory conversion made.").toStdString() << endl;	
			if (!repere && currentCarte->getDoOrtho()) getParamMain()->setAvancement(Ortho);
			else {
				getParamMain()->modifParamMicmac()[i].setACalculer(false);
				emit saveCalcul();
				getParamMain()->setAvancement(Enregistrement);
			}
		}

		if (getParamMain()->getAvancement()==Ortho && !repere && currentCarte->getDoOrtho()) {	
			//calcul orthos
			cout << tr("Orthoimage %1 computation :\n").arg(currentCarte->getImageDeReference()).toStdString();
			QString newTempoDir = QString("cd ") + noBlank(getParamMain()->getDossier()) + QString(" & ");
			QString commande = comm(newTempoDir + noBlank(getMicmacDir()) + QString("bin/MICMAC ") + noBlank(mmorthoXml) + QString(" >")+noBlank(getStdoutfile()));
			if (execute(commande)!=0) {
				getParamMain()->setAvancement(-4);
				return;
			}	
			cout << tr("Orthoimage computed.").toStdString() << endl;

			getParamMain()->modifParamMicmac()[i].setACalculer(false);
			emit saveCalcul();
			getParamMain()->setAvancement(Enregistrement);
		}
	}
	deleteFile(getParamMain()->getDossier()+getParamMain()->getCartesXML());

	cout << tr("Done").toStdString() << endl;
	getParamMain()->setAvancement(Termine);
	setProgressLabel(conv(tr("Done")));
}

bool MicmacThread::killProcess() {
	bool b = false;
	if (killall("MICMAC")==0) b = true;
	return b;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Cartes8BThread::Cartes8BThread (ParamMain* pMain, const QString& stdoutfilename, ParamConvert8B* pConvert8B, QLabel *label, bool* annul) : AppliThread(pMain,stdoutfilename,annul)
	{ paramConvert8B=pConvert8B; infoLabel=label; setEndResult(paramConvert8B->getImages().count()); }
Cartes8BThread::~Cartes8BThread() {
	delete paramConvert8B;
}

void Cartes8BThread::run() {
	getParamMain()->setAvancement(0);
	for (QList<ParamConvert8B::carte8B>::const_iterator it=paramConvert8B->getImages().begin(); it!=paramConvert8B->getImages().end(); it++) {
		//progress dialog
		if (*getAnnulation()) {
			return;
		}
		setProgressLabel(tr("Depth map conversion ")+(it->getCarte16B()));

		//masque correspondant
		QString dir = getParamMain()->getDossier() + it->getCarte16B().section("/",0,-2) + QString("/");
		if (paramConvert8B->useMasque) {
			//on vérifie que ce fichier existe toujours
			if(!QFile(dir + it->getMasque()).exists()) {
				setReaderror(dir+it->getMasque());
				getParamMain()->setAvancement(-1);			
				return;
			}
		}

		//progress dialog
		if (*getAnnulation())
			return;

		//exécution
		QString outfile = paramConvert8B->getOutFile(getParamMain()->getDossier(), *it);
		if (!paramConvert8B->getCommande(getMicmacDir(), getParamMain()->getDossier(), *it, outfile, getStdoutfile())) {
			setReaderror(it->getCarte16B());
			getParamMain()->setAvancement(-2);			
			return;
		}

		infoLabel->setText(infoLabel->text() + tr("\nDepth map %1 converted into 8 bit image.").arg(it->getCarte16B()));

		//progress dialog
		if (*getAnnulation())
			return;
		getParamMain()->setAvancement(getParamMain()->getAvancement()+1);
	}
	setProgressLabel(conv(tr("Done")));	
}

bool Cartes8BThread::killProcess() {
	bool b = false;
	if (killall("GrShade")==0) b = true;
	return b;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


LectureCamThread::LectureCamThread(int num, ParamMain* pMain, Pose* cam) : AppliThread(pMain,QString(),0), i(num), camera(cam)
        {}
LectureCamThread::~LectureCamThread () {}

void LectureCamThread::run() {
	setDone(false);	
	//param initiaux + nom du fichier
	camera->setNomImg(getParamMain()->getParamApero().getImgToOri().at(i));
	
	//nom du fichier
	QString file = getParamMain()->getDossier() + QString("Ori-F/") + QString("OrFinale-") + getParamMain()->getParamApero().getImgToOri().at(i).section(".",0,-2) + QString(".xml");
	//conversion du fichier (pb de ,)
		//récupération des fichiers appelés (calibrations...)
	QFile oldFile(file);
	if (!oldFile.open(QFile::ReadOnly | QFile::Text)) {
			setReaderror(conv(tr("Fail to read file %1.")).arg(file));
			return;
	}
	QXmlStreamReader xmlReader(&oldFile);
	QString calibXml, calibXml2;
	while (!xmlReader.atEnd() && (!xmlReader.isStartElement() || xmlReader.name().toString()!=QString("FileInterne")))
		xmlReader.readNext();
	if (!xmlReader.atEnd())
		calibXml = xmlReader.readElementText();

		//conversion du fichier de calibration
	if (!calibXml.isEmpty()) {
		calibXml2 = calibXml.section(".",0,-2) + QString("2.xml");
		if (!QFile(calibXml2).exists()) {
			QFile fxml(calibXml);
			if (!fxml.open(QIODevice::ReadOnly | QFile::Text)) {
				setReaderror(conv(tr("Fail to read file %1.")).arg(calibXml));
				return;
			} 
			QTextStream inStream(&fxml);

			QFile fxml2(calibXml2);
			if (!fxml2.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
				setReaderror(conv(tr("Fail to create file %1.")).arg(calibXml2));
				return;
			} 
			QTextStream outStream(&fxml2);

			QString text = inStream.readAll();
			text.replace(".",",");		
			outStream << text;
			fxml.close();
			fxml2.close();
		}
	}

		//conversion du fichier des poses
	QString orient = file.section(".",0,-2) + QString("2.xml");
	if (!QFile(orient).exists()) {
		QFile oldFile(file);
		if (!oldFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
			setReaderror(conv(tr("Fail to read file %1.")).arg(file));
			return;
		} 
		QTextStream inStream(&oldFile);
		QFile newFile(orient);
		if (!newFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
			setReaderror( conv(tr("Fail to create file %1.")).arg(orient));
			return;
		} 
		QTextStream outStream(&newFile);
		QString text = inStream.readAll();
		text.replace(".",",");	
		text.replace(",xml",".xml");	//cas particulier des fichiers inclus
		if (!calibXml.isEmpty()) text.replace(calibXml,calibXml2);
		outStream << text;
		oldFile.close();
		newFile.close();
	}

	//lecture
	camera->setCamera(Cam_Gen_From_File(orient.toStdString(), string("OrientationConique"), 0)->CS());
	setDone(true);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Model3DThread::Model3DThread() : AppliThread(), i(0), glParam(0) {}
Model3DThread::Model3DThread(ParamMain* pMain, GLParams*& pGL) : AppliThread(pMain,QString(),0), i(0), glParam(pGL){}
Model3DThread::Model3DThread(const Model3DThread& model3DThread) : AppliThread() { copie(this, model3DThread); }
Model3DThread::~Model3DThread () {}

void Model3DThread::setI(int num) { i=num; }

Model3DThread& Model3DThread::operator=(const Model3DThread& model3DThread) {
	if (&model3DThread!=this)
		copie(this, model3DThread);
	return *this;
}

void copie(Model3DThread* model3DThread1, const Model3DThread& model3DThread2) {
	copie(dynamic_cast<AppliThread*>(model3DThread1), model3DThread2);
	model3DThread1->i = model3DThread2.i;
	model3DThread1->glParam = model3DThread2.glParam;
}


void Model3DThread::run() {
	setDone(false);
	//numéro de l'image de référence
	QString img = getParamMain()->getParamMicmac().at(i).getImageDeReference();
	bool ok = false;
	QString num = getParamMain()->getNumImage( img, &ok, false );
	if (!ok) {
		setReaderror(conv(tr("Fail to extract referencing image %1 number.")).arg(img));
		return;
	}
	glParam->modifNuages()[i].setFromTA( !getParamMain()->getParamMicmac().at(i).getRepere() && !getParamMain()->getParamMicmac().at(i).getOrthoCalculee());

	//initialisation du nuage
	glParam->modifNuages()[i].setCarte( getParamMain()->getDossier() + img );
		//image pour la texture
		QString imgTexture;
		if (getParamMain()->getParamMicmac().at(i).getRepere()) imgTexture = getParamMain()->getDossier()+getParamMain()->convertTifName2Couleur(img);
		else if (getParamMain()->getParamMicmac().at(i).getOrthoCalculee() && QFile(getParamMain()->getDossier()+QString("ORTHO%1/Ortho-Eg-Test-Redr.tif").arg(num)).exists()) imgTexture = getParamMain()->getDossier()+QString("ORTHO%1/Ortho-Eg-Test-Redr.tif").arg(num);
		else if (getParamMain()->getParamMicmac().at(i).getOrthoCalculee() && QFile(getParamMain()->getDossier()+QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(num)).exists()) imgTexture = getParamMain()->getDossier()+QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(num);
		else  imgTexture = imgNontuilee(getParamMain()->getParamMicmac().at(i).getImageSaisie(*getParamMain()));	//TA n&b détuilé
	glParam->modifNuages()[i].setImageCouleur( imgTexture );

		//géoréférencement
	QString fichierGeoref;
	if (getParamMain()->getParamMicmac().at(i).getRepere() || getParamMain()->getParamMicmac().at(i).getOrthoCalculee()) {
		QString dir = getParamMain()->getDossier() + QString("Geo%1%2/").arg(getParamMain()->getParamMicmac().at(i).getRepere()? QString("I") : QString("Ter")).arg(num);
		fichierGeoref = dir + QString("Z_Num7_DeZoom1_Geom-Im-%1.xml").arg(num);
	} else {
		fichierGeoref = getParamMain()->getParamMicmac().at(i).getReferencementMasque(*getParamMain());
	}
	glParam->modifNuages()[i].setGeorefMNT( GeorefMNT(fichierGeoref, !getParamMain()->getParamMicmac().at(i).getRepere()) );

	if (!glParam->getNuages().at(i).getGeorefMNT().getDone()) {
		setReaderror(conv(tr("Fail to read georeferencing file %1.")).arg(fichierGeoref));
		return;
	}

	//threads
	QVector<Model3DSousThread> model3DSousThread(glParam->modifNuages().at(i).getNbResol()+1,Model3DSousThread(i, num, getParamMain(), glParam));
	for (int j=0; j<glParam->modifNuages().at(i).getNbResol()+1; j++)	//étapes : 0 à 6, nuages pour la vue 3D : 0 à 5 (le 6e = le 7e)	
		model3DSousThread[j].setJ(j+1);
	for (int j=0; j<glParam->modifNuages().at(i).getNbResol()+1; j++)
		model3DSousThread[j].start();
	for (int j=0; j<glParam->modifNuages().at(i).getNbResol()+1; j++)
		while (model3DSousThread[j].isRunning()) {}
	for (int j=0; j<glParam->modifNuages().at(i).getNbResol()+1; j++) {
		if (!model3DSousThread[j].getIsDone()) {
			setReaderror(model3DSousThread[j].getReaderror());
			return;
		}
	}

	/*QVector<Model3DSousThread> model3DSousThread(1,Model3DSousThread(i, num, getParamMain(), glParam));
	for (int j=0; j<glParam->modifNuages().at(i).getNbResol()+1; j++) {	//étapes : 0 à 6, nuages pour la vue 3D : 0 à 5 (le 6e = le 7e)	
		model3DSousThread[0].setJ(j+1);
		model3DSousThread[0].start();
		while (model3DSousThread[0].isRunning()) {}
		if (!model3DSousThread[0].getIsDone()) {
			setReaderror(model3DSousThread[0].getReaderror());
			return;
		}
	}*/

	//focale en pixel
	QString calib;
	for (int j=0; j<getParamMain()->getCorrespImgCalib().count(); j++) {
		if (getParamMain()->getCorrespImgCalib().at(j).getNumero()==num) {
			calib = getParamMain()->getCorrespImgCalib().at(j).getCalibration();
			break;
		}
	}
	if (calib.isEmpty()) {
		setReaderror(conv(tr("Fail to find calibration matching image %1.")).arg(img));
		return;
	}
	CalibCam calibCam;
	QString err = FichierCalibCam::lire(getParamMain()->getDossier(), calib, calibCam);
	if (!err.isEmpty()) {
		setReaderror(err);
		return;
	}
	glParam->modifNuages()[i].setFocale( calibCam.getFocalePx() );
	setDone(true);
}


//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//

Model3DSousThread::Model3DSousThread() : AppliThread(), i(0), j(0), numCarte(QString()), glParam(0) {}
Model3DSousThread::Model3DSousThread(int num, const QString& carte, ParamMain* pMain, GLParams*& pGL) : AppliThread(pMain,QString(),0), i(num), numCarte(carte), glParam(pGL)
        {}	//j = 1 à 7
Model3DSousThread::Model3DSousThread(const Model3DSousThread& model3DSousThread) : AppliThread() { copie(this, model3DSousThread); }
Model3DSousThread::~Model3DSousThread () {}

void Model3DSousThread::setJ(int num) { j=num; }

Model3DSousThread& Model3DSousThread::operator=(const Model3DSousThread& model3DSousThread) {
	if (&model3DSousThread!=this)
		copie(this, model3DSousThread);
	return *this;
}

void copie(Model3DSousThread* model3DSousThread1, const Model3DSousThread& model3DSousThread2) {
	copie(dynamic_cast<AppliThread*>(model3DSousThread1), model3DSousThread2);
	model3DSousThread1->i = model3DSousThread2.i;
	model3DSousThread1->j = model3DSousThread2.j;
	model3DSousThread1->numCarte = model3DSousThread2.numCarte;
	model3DSousThread1->glParam = model3DSousThread2.glParam;
}

void Model3DSousThread::run() {
	setDone(false);
	QString dir = getParamMain()->getDossier() + QString("Geo%1").arg(getParamMain()->getParamMicmac().at(i).getRepere()? QString("I") : QString("Ter")) + numCarte + QString("/");

	QString file = dir + QString("NuageImProf_Geom-Im-%1_Etape_%2.xml").arg(numCarte).arg(j);
	if (!QFile(file).exists()) {
		setReaderror(conv(tr("Point cloud %1 not found.")).arg(file));
		return;
	}

	//conversion pour lecture (, -> .)
	if (j==7) { //l'étape 7 == l'étape 6 lissée, même dézoom : on ne l'affiche pas
	    setDone(true);
	    return;
	}
	QString virgule;
	QString err = systemeNumerique(virgule);
	if (!err.isEmpty()) {
		setReaderror(err);
		return;
	} 

	QString tempo;
	if (virgule==QString(",")) {
		tempo = dir + QString("tempoNuageImProf_Geom-Im-%1_Etape_%2.xml").arg(numCarte).arg(j);
		QFile oldFile(file);
		if (!oldFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
			setReaderror(conv(tr("Fail to read file %1.ply.")).arg(file.section(".",0,-2)));
			return;
		} 
		QTextStream inStream(&oldFile);
		QFile newFile(tempo);
		if (!newFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
			setReaderror(conv(tr("Fail to create temporary file %1.")).arg(tempo));
			return;
		} 
		QTextStream outStream(&newFile);
		QString text = inStream.readAll();
		text.replace(".",",");	
		text.replace(",tif",".tif");	
		outStream << text;
		oldFile.close();
		newFile.close();
	}

	QString fichier = (virgule==QString(","))? tempo : file;
	cElNuage3DMaille* nuage = cElNuage3DMaille::FromFileIm(comm(fichier).toStdString());
	glParam->modifNuages()[i].modifPoints()[j-1] = nuage;
	glParam->modifNuages()[i].modifCorrelation()[j-1] = dir+QString("Correl_Geom-Im-%1_Num_%2.tif").arg(numCarte).arg(j);
	if (virgule==QString(",")) QFile(tempo).remove();
	setDone(true);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


OrthoThread::OrthoThread():
	AppliThread(),
	infoLabel(0),
	numRef(QString()),
	egaliser(false) {}
OrthoThread::OrthoThread(ParamMain* pMain, QLabel *label, bool egal):
	AppliThread(pMain,QString(),0),
	infoLabel(label),
	numRef(QString()),
	egaliser(egal){}
OrthoThread::OrthoThread(const OrthoThread& orthoThread) : AppliThread() { copie(this, orthoThread); }
OrthoThread::~OrthoThread () {}

void OrthoThread::setNumImgRef(QString num) { numRef=num; }

OrthoThread& OrthoThread::operator=(const OrthoThread& orthoThread) {
	if (&orthoThread!=this)
		copie(this, orthoThread);
	return *this;
}

void copie(OrthoThread* orthoThread1, const OrthoThread& orthoThread2) {
	copie(dynamic_cast<AppliThread*>(orthoThread1), orthoThread2);
	orthoThread1->numRef = orthoThread2.numRef;
	orthoThread1->infoLabel = orthoThread2.infoLabel;
	orthoThread1->egaliser = orthoThread2.egaliser;
}

void OrthoThread::run() {
	setDone(false);

	//fichiers xml
	QString dossier = getParamMain()->getDossier() + QString("ORTHO%1/").arg(numRef);
	deleteFile(dossier+QString("Porto.xml"));
	if (!egaliser) QFile(getParamMain()->getMicmacDir()+"/interface/xml/Porto.xml").copy(dossier+QString("Porto.xml"));
	else QFile(getParamMain()->getMicmacDir()+"/interface/xml/PortoEgal.xml").copy(dossier+QString("Porto.xml"));
	if (!FichierPorto::ecrire(dossier+getParamMain()->getPathMntXML(), numRef)) {
		setReaderror(conv(tr("Fail to write file %1.")).arg(dossier+getParamMain()->getPathMntXML()));
		return;
	}
	deleteFile(dossier+QString("Label-Test-Redr.tif"));

	//calcul
	QString commande = comm(QString("%1bin/Porto %2Porto.xml").arg(getMicmacDir()).arg(dossier));
	if (execute(commande)!=0) {
		setReaderror(conv(tr("Fail to compute mosaic of orthoimage ORTHO%1/.")).arg(numRef));
		return;
	}

	infoLabel->setText(infoLabel->text() + conv(tr("\nOrthoimage ORTHO%1/ mosaic computed.")).arg(numRef));
	setDone(true);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Convert2PlyThread::Convert2PlyThread() : AppliThread(), paramNuage(0), infoLabel(0) {}
Convert2PlyThread::Convert2PlyThread(ParamMain* pMain, QLabel *label, const ParamPly& param) : AppliThread(pMain,QString(),0), paramNuage(0), infoLabel(label), paramPly(&param)
        {}
Convert2PlyThread::Convert2PlyThread(const Convert2PlyThread& convert2PlyThread) : AppliThread() { copie(this, convert2PlyThread); }
Convert2PlyThread::~Convert2PlyThread () {}

void Convert2PlyThread::setParamNuage(const ParamNuages* param) { paramNuage=param; }

Convert2PlyThread& Convert2PlyThread::operator=(const Convert2PlyThread& convert2PlyThread) {
	if (&convert2PlyThread!=this)
		copie(this, convert2PlyThread);
	return *this;
}

void copie(Convert2PlyThread* convert2PlyThread1, const Convert2PlyThread& convert2PlyThread2) {
	copie(dynamic_cast<AppliThread*>(convert2PlyThread1), convert2PlyThread2);
	convert2PlyThread1->paramNuage = convert2PlyThread2.paramNuage;
	convert2PlyThread1->paramPly = convert2PlyThread2.paramPly;
	convert2PlyThread1->infoLabel = convert2PlyThread2.infoLabel;
}

void Convert2PlyThread::run() {
	setDone(false);
	QString fichierPly = paramNuage->getFichierPly();
	QString fichierXml = paramNuage->getFichierXml();
	if (QFile(fichierPly).exists()) {
		cout << tr("File %1 already exists.").arg(fichierPly).toStdString() << endl;
		setDone(true);
		return;
	}
	//filtrage des points bruités
	QString masqueFiltre;
	if (paramPly->getDoFiltrage()) {
		QString commande = paramNuage->commandeFiltrage(masqueFiltre);
cout << "masque filtre " << masqueFiltre.toStdString() << endl;
		if (commande.isEmpty()) {
			setReaderror(conv(tr("Fail to read xml file %1 ; no filtering is possible.")).arg(fichierXml));
			return;
		}
		if (!QFile(masqueFiltre).exists()) {
			if (execute(commande)!=0) {
				deleteFile(masqueFiltre);
				setReaderror(conv(tr("Fail to filter cloud %1.")).arg(fichierXml));
				return;
			}
		}
	}
	//conversion
	QString commande;
	QString err = paramNuage->commandePly(commande, getMicmacDir(), *paramPly, masqueFiltre, *getParamMain());
	if (!err.isEmpty()) {
		setReaderror(err);
		return;
	}
	if (commande.isEmpty()) {
		setReaderror(conv(tr("Fail to rescale filtered cloud mask, no possible conversion.")).arg(fichierXml));
		return;
	}
	if (execute(commande)!=0) {
		setReaderror(conv(tr("Fail to convert point cloud %1 into ply format.")).arg(fichierXml));
		return;
	}
	//nettoyage
	deleteFile(masqueFiltre.section(".",0,-2) + QString("tempo.tif"));
	for (int i=0; i<paramPly->getMasques().count(); i++)
		deleteFile(paramPly->getMasques().at(i).second.section(".",0,-2) + QString("tempo.tif"));

	infoLabel->setText(infoLabel->text() + tr("\nPoint cloud %1 converted into ply format.").arg(paramNuage->getFichierXml().section("/",-2,-1)));
	setDone(true);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


NuageThread::NuageThread(GLWidget* glWidg, int num, int dezoom) : AppliThread(0,QString(),0), i(num), j(dezoom), glWidget(glWidg)
        {}
NuageThread::~NuageThread () {}

GLuint& NuageThread::getObjectNuag() { return nuageobj; }

void NuageThread::run() { nuageobj = glWidget->makeObjectNuag(i,j); }
