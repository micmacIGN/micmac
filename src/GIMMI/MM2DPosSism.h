#ifndef MM2DPOSSIM
#define MM2DPOSSIM

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

class MM2DPosSism : public QWidget
{

    Q_OBJECT


public:
    MM2DPosSism(QList <QString>, QList <QString>, QString);
    QList<QString> listImages;
    QMessageBox msgBox2;
    QString msg1();
    QDialog qDi;




signals:

public slots:
    void mm3d();
    void msg();
    void readyReadStandardOutput();
    void oriSearch2();
    void oriSearch1();
    void stopped();



private:
    QButtonGroup *modeGroupBox;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    Console cons;
    QString sendCons;
    QTextEdit *Qtr ;

    QRadioButton *mulScale;
    QRadioButton *all;
    QRadioButton *line;
    QRadioButton *file;
    QRadioButton *graph;
    QRadioButton *georeph;
    QFileInfo fichier;

    std::string mode;
    std::string images;
    std::string dirMecToStdStrg;
    QTextEdit *dirMec;
    QTextEdit *resoM;
    std::string resom_str;
    std::string resoM_str;
    QCheckBox* boxXml;
    std::string txt;
    QTextEdit *labelImageTxt1;
    QTextEdit *labelImageTxt2;
    QString aPath;



    std::string cmd;

    std::string path_s;
    QString p_stdout;
    QProcess p;
    std::string recup;


};





#endif // MM2DPOSSIM

