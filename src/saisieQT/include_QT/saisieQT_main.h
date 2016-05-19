#ifndef SAISIEMASQQT_MAIN_H
#define SAISIEMASQQT_MAIN_H

#include "QT_interface_Elise.h"

using namespace std;

void updateSettings(QSettings &settings, Pt2di aSzWin, Pt2di aNbFen, bool aForceGray);

extern void SaisieAppuisInit(int argc, char ** argv,
							 Pt2di &aSzW,
							 Pt2di &aNbFen,
                             std::string &aFullName,
                             std::string &aDir,
                             std::string &aName,
                             std::string &aNamePt,
                             std::string &anOri,
                             std::string &aModeOri,
                             std::string &anOut,
                             std::string &aNameAuto,
                             std::string &aPrefix2Add,
                             bool &aForceGray,
                             double &aZMoy,
                             double &aZInc);

extern void SaisieAppuisPredic(int argc, char ** argv,
							   Pt2di &aSzW,
							   Pt2di &aNbFen,
                               std::string &aFullName,
                               std::string &aDir,
                               std::string &aName,
                               std::string &aNamePt,
                               std::string &anOri,
                               std::string &aModeOri,
                               std::string &aNameMesure,
                               std::string &aTypePts,
                               double &aFlou,
                               bool &aForceGray,
                               double &aZMoy,
                               double &aZInc);

extern void SaisieBasc(int argc, char ** argv,
                        std::string &aFullName,
                        std::string &aDir,
                        std::string &aName,
                        std::string &anOri,
                        std::string &anOut,
						Pt2di &aSzW,
						Pt2di &aNbFen,
                        bool &aForceGray);

QApplication & getQApplication();

int helpMessage(QApplication const &app, QString text);
bool checkNamePt(QString text);
QStringList getFilenames(std::string aDir, std::string aName);

void loadTranslation(QApplication &app);

int saisieMasqQT_main(int argc, char *argv[]);
int saisieAppuisInitQT_main(int argc, char *argv[]);
int saisieAppuisPredicQT_main(int argc, char *argv[]);
int saisieBoxQT_main(int argc, char *argv[]);
int saisieBascQT_main(int argc, char *argv[]);

void saisieMasq_ElInitArgMain(int argc, char ** argv, std::string &aFullName, std::string &aPost, std::string &aNameMasq, std::string &aAttr, Pt2di &aSzW, double &aGama);

int qtpopup(const std::string &aText);

#define QTPOPUP(text){\
	std::ostringstream ss;\
	ss << text << std::endl;\
	qtpopup(ss.str());\
}

#endif
