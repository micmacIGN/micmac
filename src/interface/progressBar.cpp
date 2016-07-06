#include "progressBar.h"

using namespace std;


Progression::Progression() : QThread() {}//problème de modification du paramMain et de progressDialog (erreur de segmentation)
Progression::~Progression() {}

void Progression::updatePercent(const ParamMain& paramMain, QProgressDialog* progressdialog, const QString& stdoutfile) const {
//met le bon pourcentage à la progressDialog en fonction du thread traitée
	int n = dispatch(paramMain, progressdialog->value(), stdoutfile);
	int n0 = lastComplete(paramMain);
	if (progressdialog->value()<n+n0) progressdialog->setValue(n+n0);
}

int Progression::dispatch(const ParamMain& paramMain, int pDialogVal, const QString& stdoutfile) const {
//recherche le pourcentage d'avancement de la commande en cours ; pour cela, lance une fonction différente en fonction du thread et de la commande traitée
	switch (paramMain.getCurrentMode()) {
		case ParamMain::PointsEnCours : 
			switch (paramMain.getAvancement()) {
				case PastisThread::Enregistrement : return 0;	//enregistrement des paramètres
				case PastisThread::Conversion : return whichDcraw(paramMain, pDialogVal);	//conversion au format tif
				case PastisThread::Ecriture : return 0;		//écriture des fichiers
				case PastisThread::FiltrCpls : return whichPastis(paramMain)/10;		//Pastis
				case PastisThread::PtsInteret : return whichPastis(paramMain)/10;		//Pastis2
				case PastisThread::ImgsOrientables : return 0;		//tri des images
				default : return 0;
			}
			break;
		case ParamMain::PoseEnCours : 
			switch (paramMain.getAvancement()) {
				case 0 : return 0;	//enregistrement des paramètres
				case 1 : return 0;	//filtrage des points homologues (test_ISA0)
				case 2 : return (paramMain.getParamApero().getAutoCalib().count()>0 )? whichApero(stdoutfile) : 0;	//auto-calibration
				case 3 : return whichApero(stdoutfile);		//Apero
				case 4 : return (paramMain.getParamApero().getMultiechelle() )? whichApero(stdoutfile) : 0;		//Apero 2ème étape
				case 5 : return (paramMain.getParamApero().getLiberCalib().count()>0)? whichApero(stdoutfile) : 0;		//Apero 2ème étape
				case 6 : return whichHomol3D(paramMain);		//enregistrement des points homologues en 3D
				default : return 0;
			}
			break;
		case ParamMain::CarteEnCours :
			switch (paramMain.getAvancement()) {
				case 0 : return 0;	//enregistrement des paramètres
				case 1 : return whichMicmac(stdoutfile);	//micmac
				case 2 : return doOrthoCurrentCarte(paramMain)? whichMicmac(stdoutfile) : 0;	//ortho
				default : return 0;
			}
			break;
		case ParamMain::EndMode : 
			return whichGrShade(stdoutfile);	//GrShade
		default : 
			return 0;
	}
	return 0;
}

int Progression::lastComplete(const ParamMain& paramMain) const {
//renvoie le pourcentage correspondant à la complétude des commandes précédentes du thread
	int n = 0;
	switch (paramMain.getCurrentMode()) {
		case ParamMain::PointsEnCours : 
			switch (paramMain.getAvancement()) {
				case PastisThread::Termine : n += 1;
				case PastisThread::ImgsOrientables : n += 2*paramMain.getParamPastis().getCouples().count()/10;
				case PastisThread::PtsInteret : n += (paramMain.getParamPastis().getMultiScale()) ? paramMain.getCorrespImgCalib().count()*(paramMain.getCorrespImgCalib().count()-1)/10 : 0;	//inconnue en fait
				case PastisThread::FiltrCpls : n += 1;
				case PastisThread::Ecriture : n += paramMain.getCorrespImgCalib().count();
				case PastisThread::Conversion : n += 1;
				case PastisThread::Enregistrement : n += 0;
					break;
				default : return 0;
			}
			break;
		case ParamMain::PoseEnCours : 
			switch (paramMain.getAvancement()) {
				case 7 : n += 2+paramMain.getParamApero().getImgToOri() .count();
				case 6 : n += (paramMain.getParamApero().getLiberCalib().count()>0)? 13 : 0;
				case 5 : n += (paramMain.getParamApero().getMultiechelle() )? 13 : 0;
				case 4 : n += 13;
				case 3 : n += (paramMain.getParamApero().getAutoCalib().count()>0 )? 13 : 0;
				case 2 : n += (paramMain.getParamApero().getFiltrage() )? 1 : 0;
				case 1 : n += 1;
				case 0 : n += 0;
					break;
				default : return 0;
			}
			break;
		case ParamMain::CarteEnCours : 
			switch (paramMain.getAvancement()) {
				case 2 : n += 59;
				case 1 : n += 1;
				case 0 : n += 0;
					break;
				default : return 0;
			}
			break;
		case ParamMain::EndMode :  {
				n += paramMain.getAvancement() * 40;	//en fait avancement * 2 * NbDir;
				break;
			}
			break;
		default : 
			return 0;
	}
	return n;
}

int Progression::maxComplete(const ParamMain& paramMain) const {
//renvoie le pourcentage correspondant à la complétude de toutes les commandes du thread
	int n = 0;
	switch (paramMain.getCurrentMode()) {
		case ParamMain::PointsEnCours : 
				n += 1;
				n += 2*paramMain.getParamPastis().getCouples().count()/10;
				n += (paramMain.getParamPastis().getMultiScale()) ? paramMain.getCorrespImgCalib().count()*(paramMain.getCorrespImgCalib().count()-1)/10 : 0;	//inconnue en fait
				n += 1;
				n += paramMain.getCorrespImgCalib().count();
				n += 1;
				n += 0;
				break;
		case ParamMain::PoseEnCours : 
				n += 2+paramMain.getParamApero().getImgToOri() .count();
				n += (paramMain.getParamApero().getLiberCalib().count()>0)? 13 : 0;
				n += (paramMain.getParamApero().getMultiechelle() )? 13 : 0;
				n += 13;
				n += (paramMain.getParamApero().getAutoCalib().count()>0 )? 13 : 0;
				n += (paramMain.getParamApero().getFiltrage() )? 1 : 0;
				n += 1;
				n += 0;
				break;
		case ParamMain::CarteEnCours : 
				n += doOrthoCurrentCarte(paramMain)? 59 : 0;
				n += 59;
				n += 1;
				n += 0;
				break;
		default : 
			return 0;
	}
	return n;
}

int Progression::maxComplete(const ParamConvert8B& paramConvert8B) const { return paramConvert8B.getImages().count() * 40; }	//en fait count * 2 * NbDir;

int Progression::whichDcraw(const ParamMain& paramMain, int pDialogVal) const {
//renvoie le pourcentage d'avancement de conversion ; pour cela, recherche si les fichiers correspondants ont été créés
//1 fichier = 1%
//on ne considère pas le images couleurs car les images tif->tif sont créées après (par lien)
	int i = max(pDialogVal - lastComplete(paramMain), 0);
	int n = 0;
	while (i<paramMain.getCorrespImgCalib().count()) {
		if (QFile(paramMain.getDossier()+paramMain.getCorrespImgCalib().at(i).getImageTif()).exists())
			i++;
		else {
			n = i;
			break;
		}		
	}
	return n;
}

int Progression::whichPastis(const ParamMain& paramMain) const {
//renvoie le pourcentage d'avancement de la commande Pastis ; pour cela, recherche si les fichiers correspondants ont été créés
//1 fichier = 0.1%
/*	cTplValGesInit<string>  aTpl;
	char** argv = new char*[1];
	argv[0] = "rthsrth";
	cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, paramMain.getDossier().toStdString(), aTpl );
	const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");	//compte le nombre de fichiers
	delete [] argv;
	delete mICNM;
	return aVN->size();*/

	QStringList l = QDir(paramMain.getDossier()).entryList(QDir::Dirs);
	int nb = 0;
	for (QStringList::const_iterator it=l.begin(); it!=l.end(); it++) {
		if (it->left(1)==QString(".")) continue;
		nb += max(0,QDir(paramMain.getDossier()+*it).entryList(QDir::Files).count());
	}
	return nb;
}

int Progression::whichApero(const QString& stdoutfile) const {
//renvoie le pourcentage d'avancement de la commande Apero ; pour cela, recherche l'étape courante dans l'outstream
//1 étape = 1%
	QFile file(stdoutfile);
	if (!file.open(QIODevice::ReadOnly))
		return 0;
	QTextStream inStream(&file);
	QString text = inStream.readAll();
	int n = text.count("End Iter");
	file.close();
	return n;
}

int Progression::whichHomol3D(const ParamMain& paramMain) const {
//renvoie le pourcentage d'avancement de l'enregistrement des points homologues en 3D
	return paramMain.getEtape();
}

int Progression::whichMicmac(const QString& stdoutfile) const {
//renvoie le pourcentage d'avancement de la commande MICMAC ; pour cela, recherche l'étape courante dans l'outstream
	QFile file(stdoutfile);
	if (!file.open(QIODevice::ReadOnly))
		return 0;
	QTextStream inStream(&file);
	QString text = inStream.readAll();
	int n = text.count("BEGIN ETAPE");
	file.close();
	return n;
}

bool Progression::doOrthoCurrentCarte(const ParamMain& paramMain) const {
	for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
		if (!paramMain.getParamMicmac().at(i).getACalculer()) continue;
		return (!paramMain.getParamMicmac().at(i).getRepere() && paramMain.getParamMicmac().at(i).getDoOrtho());
	}
	return false; // ?
}

int Progression::whichGrShade(const QString& stdoutfile) const {
//renvoie le pourcentage d'avancement de la commande MICMAC ; pour cela, recherche l'étape courante dans l'outstream
	QFile file(stdoutfile);
	if (!file.open(QIODevice::ReadOnly))
		return 0;
	QTextStream inStream(&file);
	QString text = inStream.readAll();
	int n = text.count("Dir");
	file.close();
	return n;
}




