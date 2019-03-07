#ifndef TARAMA
#define TARAMA

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


class Tarama : public QWidget
{

    Q_OBJECT


public:
    Tarama(QList <QString>,QList <QString>, QList <QString>, QString);
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
    QLCDNumber *m_lcd;
    QButtonGroup *groupBox;
    QCheckBox* box;
    QList<QCheckBox*> listeCases;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
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
    QSlider *m_slider;
    QComboBox *groupBox1;
    QProcess p;
    QString p_stdout;
    Console cons;
    QString sendCons;


};

#endif // TARAMA

