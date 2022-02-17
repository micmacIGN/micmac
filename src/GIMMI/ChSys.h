#ifndef CHSYS
#define CHSYS



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
#include <QProcess>
#include "MainWindow.h"
#include "NewProject.h"
#include "Tapioca.h"
#include "Console.h"

class ChSys : public QWidget
{

    Q_OBJECT


public:
    ChSys(QString);
    QList<QString> listImages;
    QMessageBox msgBox2;
    QString msg1();




signals:

public slots:
    void mm3d();



private:
    QButtonGroup *modeGroupBox;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    Console cons;
    QString sendCons;
    QTextEdit *rPC;
    QFileInfo fichier;

    QRadioButton *mulScale;
    QRadioButton *all;
    QRadioButton *line;
    QRadioButton *file;
    QRadioButton *graph;
    QRadioButton *georeph;

    std::string mode;
    std::string images;
    QTextEdit *resom;
    QTextEdit *resoM;
    std::string resom_str;
    std::string resoM_str;
    QCheckBox* boxXml;
    std::string txt;
    QCheckBox* boxSouth;
    std::string south;
    QTextEdit *utmQt;
    std::string utm;
    std::string chemin;
    std::string path_s;
    QString p_stdout;
    QProcess p;
    std::string recup;


};




#endif // CHSYS

