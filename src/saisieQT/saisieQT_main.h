#ifndef SAISIEMASQQT_MAIN_H
#define SAISIEMASQQT_MAIN_H

#include <QtGui>
#include <QApplication>
#include "mainwindow.h"

extern void SaisieAppuisInit(int argc, char ** argv,
                                  Pt2di &aSzW,
                                  Pt2di &aNbFen,
                                  std::string &aFullName,
                                  std::string &aDir,
                                  std::string &aName,
                                  std::string &aNamePt,
                                  std::string &anOri,
                                  std::string &anOut,
                                  std::string &aNameAuto,
                                  std::string &aPrefix2Add,
                                  bool &aForceGray);

int helpMessage(QApplication const &app, QString text);

int saisieMasqQT_main(QApplication &app);
int saisieAppuisInitQT_main(QApplication &app, int argc, char *argv[]);

#endif
