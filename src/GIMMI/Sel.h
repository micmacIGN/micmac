#ifndef SEL_H
#define SEL_H



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

class Sel : public QWidget
{

    Q_OBJECT


public:
    Sel(QList <QString>, QList <QString>, QString);
    QList<QString> listImages;
    QMessageBox msgBox2;
    QString msg1();




signals:

public slots:
    void mm3d();
    void msg();




private:
    QButtonGroup *modeGroupBox;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    Console cons;
    QString sendCons;

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

    std::string cmd;

    std::string path_s;
    QString p_stdout;
    QProcess p;
    std::string recup;


};

#endif // SEL_H
