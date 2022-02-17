#ifndef C3DC
#define C3DC






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

#include "MainWindow.h"

#include "NewProject.h"
#include "Tapioca.h"
#include "Campari.h"
#include "Console.h"


class C3dc : public QWidget
{


    Q_OBJECT


public:
    C3dc(QList <QString>, QList <QString>, QList <QString>, QString);
    QList<QString> list2;
    QList<QString> listImages;
    QString msg1();
    QMessageBox msgBox2;
    QDialog qDi;



signals:
    void signal_c();

public slots:
    void mm3d();
    QString getTxt();
    void msg();
    void rpcFile();
    void readyReadStandardOutput();
    void stopped();



private:
    QCheckBox* box;
    QList<QCheckBox*> listeCases;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    QButtonGroup group;
    QTextEdit *outPutCampT;
    QFileInfo fichier;
    QTextEdit *Qtr ;
    QRadioButton *Ortho;
    QRadioButton *UrbanMNE;
    QRadioButton *GeomImage;
    std::string mode;
    std::string modo;
    std::string images;
    std::string outPutCampStd;
    std::string sizeM;
    QTextEdit *resoM;
    QCheckBox* boxTxt ;
    std::string txt;
    std::string cmd;
    std::string path_s;
    QComboBox *groupBox1;
    QComboBox *groupBox2;
    QProcess p;
    QString p_stdout;
    Console cons;
    QString sendCons;



    QTextEdit *Masq3DQt;
    QTextEdit *OutQt;
    QTextEdit *SzNormQt;
    QCheckBox *PlyCoulQt;
    QCheckBox *TuningQt;
    QCheckBox *PurgeQt;
    QTextEdit *DownScaleQt;
    QTextEdit *ZoomFQt;
    QCheckBox *UseGpuQt;
    QTextEdit *DefCorQt;
    QTextEdit *ZRegQt;
    QCheckBox *ExpTxtQt;
    QTextEdit *FilePairQt;
    QCheckBox *DebugMMByPQt;
    QCheckBox *BinQt;
    QCheckBox *ExpImSecQt;
    QTextEdit *OffsetPlyQt;
    QTextEdit *OffsetPlyQt2;
    QTextEdit *OffsetPlyQt3;


    std::string Masq3DVar;
    std::string OutVar;
    std::string SzNormVar;
    std::string PlyCoulVar;
    std::string TuningVar;
    std::string PurgeVar;
    std::string DownScaleVar;
    std::string ZoomFVar;
    std::string UseGpuVar;
    std::string DefCorVar;
    std::string ZRegVar;
    std::string ExpTxtVar;
    std::string FilePairVar;
    std::string DebugMMByPVar;
    std::string BinVar;
    std::string ExpImSecVar;
    std::string OffsetPlyVar;
    std::string OffsetPlyVar2;
    std::string OffsetPlyVar3;



};




#endif // C3DC

