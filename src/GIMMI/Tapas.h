#ifndef TAPAS_H
#define TAPAS_H



#include <QWidget>
#include <QMenuBar>
#include <QLineEdit>
#include <QLayout>
#include <QToolBar>
#include <QFormLayout>
#include <QDockWidget>
#include <QTextEdit>
#include <QListWidget>
#include <QLabel>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QFileSystemModel>
#include <iostream>
#include <QModelIndex>
#include <QStringListModel>
#include <stdio.h>
#include <dirent.h>
#include <regex.h>
#include <string>
#include <QDebug>
#include <QDirModel>
#include <QLCDNumber>
#include <QSlider>
#include <QDockWidget>
#include <QPushButton>
#include <QMainWindow>
#include <QButtonGroup>
#include <QComboBox>

#include "MainWindow.h"

#include "NewProject.h"
#include "Tapioca.h"
#include "Tapas.h"
#include "Console.h"



class Tapas : public QWidget
{

    Q_OBJECT


public:
    Tapas(QList <QString>, QList <QString>, QString);
    QList<QString> list2;
    QList<QString> listImages;
    QString msg1();
    QMessageBox msgBox2;
    QDialog qDi;
    std::string sendOri();



signals:
    void signal_b();


public slots:
    void mm3d();
    QString getTxt();
    void msg();
    void readyReadStandardOutput();
    void stopped();


private:
    QComboBox* groupBox;
    QTextEdit *Qtr ;
    QCheckBox* imagesCheckBox;


    QTextEdit *OutQt;
    QTextEdit *inOriQt;
    QTextEdit *inCalQt;
    QTextEdit *ExpTxtQt;
    QTextEdit *DoCQt;
    QTextEdit *ForCalibQt;
    QTextEdit *FocsQt;
    QTextEdit *FocsQt2;
    QTextEdit *VitesseInitQt;
    QTextEdit *PPRelQt;
    QTextEdit *PPRelQt2;
    QTextEdit *DecentreQt;
    QTextEdit *PropDiagQt;
    QTextEdit *SauvAutomQt;
    QTextEdit *ImInitQt;
    QCheckBox *MOIQt;
    QTextEdit *DBFQt;
    QCheckBox *DebugQt;
    QTextEdit *DegRadMaxQt;
    QTextEdit *DegGenQt;
    QCheckBox *LibAffQt;
    QCheckBox *LibDecQt;
    QCheckBox *LibPPQt;
    QCheckBox *LibCPQt;
    QCheckBox *LibFocQt;
    QTextEdit *RapTxtQt;
    QTextEdit *LinkPPaPPsQt;
    QTextEdit *FrozenPosesQt;
    QTextEdit *FrozenCentersQt;
    QTextEdit *FrozenOrientsQt;
    QCheckBox *FreeCalibInitQt;
    QTextEdit *FrozenCalibsQt;
    QTextEdit *FreeCalibsQt;
    QTextEdit *SHQt;
    QCheckBox *RefineAllQt;
    QTextEdit *ImMinMaxQt;
    QTextEdit *EcMaxQt;
    QTextEdit *EcInitQt;
    QTextEdit *EcInitQt2;
    QTextEdit *CondMaxPanoQt;
    QTextEdit *SinglePosQt;
    QTextEdit *RankInitFQt;
    QTextEdit *RankInitPPQt;
    QTextEdit *RegulDistQt;
    QTextEdit *MulLVMQt;
    QCheckBox *MultipleBlockQt;


    std::string OutVar;
    std::string inOriVar;
    std::string inCalVar;
    std::string ExpTxtVar;
    std::string DoCVar;
    std::string ForCalibVar;
    std::string FocsVar;
    std::string Focs2Var;
    std::string VitesseInitVar;
    std::string PPRelVar;
    std::string PPRel2Var;
    std::string DecentreVar;
    std::string PropDiagVar;
    std::string SauvAutomVar;
    std::string ImInitVar;
    std::string MOIVar;
    std::string DBFVar;
    std::string DebugVar;
    std::string DegRadMaxVar;
    std::string DegGenVar;
    std::string LibAffVar;
    std::string LibDecVar;
    std::string LibPPVar;
    std::string LibCPVar;
    std::string LibFocVar;
    std::string RapTxtVar;
    std::string LinkPPaPPsVar;
    std::string FrozenPosesVar;
    std::string FrozenCentersVar;
    std::string FrozenOrientsVar;
    std::string FreeCalibInitVar;
    std::string FrozenCalibsVar;
    std::string FreeCalibsVar;
    std::string SHVar;
    std::string RefineAllVar;
    std::string ImMinMaxVar;
    std::string EcMaxVar;
    std::string EcInitVar;
    std::string EcInit2Var;
    std::string CondMaxPanoVar;
    std::string SinglePosVar;
    std::string RankInitFVar;
    std::string RankInitPPVar;
    std::string RegulDistVar;
    std::string MulLVMVar;
    std::string MultipleBlockVar;

    QList<QCheckBox*> listeCasesImages;
    QTextEdit *outS;
    std::string mode;
    std::string images;
    std::string out;
    QCheckBox* boxTxt ;
    QCheckBox* boxRefineAll;
    std::string txt;
    std::string refineAll;




    std::string cmd;
    std::string path_s;
    Console cons;
    QString sendCons;
    QProcess p;
    QString p_stdout;


};

#endif // TAPAS_H
