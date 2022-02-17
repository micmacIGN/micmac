#ifndef CONVERT2GENBUNDDLE
#define CONVERT2GENBUNDDLE



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


class Convert2GenBundle : public QWidget
{

    Q_OBJECT


public:
    Convert2GenBundle(QList <QString>,QList <QString>, QList <QString>, QString);
    QList<QString> list2;
    QList<QString> list3;
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
    QLCDNumber *m_lcd;
    QButtonGroup *groupBox;
    QCheckBox* box;
    QList<QCheckBox*> listeCases;
    QList<QRadioButton*> imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    QTextEdit *outPutCampT;
    QFileInfo fichier;
    QTextEdit *Qtr ;
    QRadioButton *Ortho;
    QRadioButton *UrbanMNE;
    QRadioButton *GeomImage;
    std::string mode;
    std::string modo1;
    std::string modo2;
    std::string modo3;
    std::string images;
    std::string outPutCampStd;
    std::string sizeM;
    QTextEdit *resoM;
    QCheckBox* boxTxt ;
    std::string txt;
    std::string cmd;
    std::string path_s;
    QSlider *m_slider;
    QComboBox *groupBox1;
    QComboBox *groupBox2;
    QComboBox *groupBox3;
    QComboBox *groupBox4;
    QComboBox *groupBox5;

    QProcess p;
    QString p_stdout;
    Console cons;
    QString sendCons;

    QTextEdit *labelDegreQt;
    QTextEdit *labelAutreQt;

    QTextEdit *ChSysQt;
    QTextEdit *DegreQt;
    QTextEdit *TypeQt;
    QTextEdit *PertubAngQt;
    QTextEdit *outPut;

      std::string ChSysVar;
      std::string DegreVar;
      std::string TypeVar;
      std::string PertubAngVar;



    std::string  labelDegreVar;
    std::string  labelAutreVar;
};





#endif // CONVERT2GENBUNDDLE

