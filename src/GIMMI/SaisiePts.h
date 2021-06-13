#ifndef SAISIEPTS_H
#define SAISIEPTS_H




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
#include "SaisiePts.h"


class SaisiePts : public QWidget
{

    Q_OBJECT


public:
    SaisiePts(QList <QString>, QList <QString>, QList <QString>, QString);
    QList<QString> list2;
    QList<QString> list3;
    QList<QString> list4;
    QList<QString> listImages;
    QString msg1();
    QMessageBox msgBox2;
    QDialog qDi;



signals:
    void signal_d();

public slots:
    void mm3d();
    QString getTxt();
    void msg();
    void stopped();
   void rpcFile();
    void readyReadStandardOutput();

private:
    QLCDNumber *m_lcd;
    QButtonGroup *groupBox;
    QCheckBox* box;
    QList<QCheckBox*> listeCases;
    QList<QCheckBox*> listeCasesImages;
    std::string modo;
    QTextEdit *Qtr ;

    QTextEdit *rPC;
    QTextEdit *outPut;
    QFileInfo fichier;
    QComboBox *groupBox1;
    QComboBox *groupBox2;
    QComboBox *groupBox3;

    QRadioButton *MulScale;
    QRadioButton *All;
    QRadioButton *Line;
    QRadioButton *File;
    QRadioButton *Graph;
    QRadioButton *Georeph;
    std::string mode;
    std::string images;
    std::string outPutStd;
    std::string oriFile;
    std::string Xml;
    std::string pFName;

    QCheckBox* boxTxt ;
    std::string txt;
    std::string cmd;
    std::string path_s;
    QString p_stdout;
    QProcess p;
    QSlider *m_slider;
    Console cons;
    QCheckBox* imagesCheckBox;
    QString sendCons;
};


#endif // SAISIEPTS_H
