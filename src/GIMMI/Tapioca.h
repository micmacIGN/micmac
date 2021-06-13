#ifndef TAPIOCA_H
#define TAPIOCA_H



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

class Tapioca : public QWidget
{

    Q_OBJECT


public:
    Tapioca(QList <QString>, QList <QString>, QString);
    QList<QString> listImages;
    QMessageBox msgBox2;
    QString msg1();
    QDialog qDi;




signals:
    void signal_a();

public slots:
    void mm3d();
    void mulScaleC();
    void allC();
    void lineC();
    void fileC();
    void graphC();
    void georephC();
    void stopped();
    QString getTxt();
    void msg();

void readyReadStandardOutput();


private:
    QButtonGroup *modeGroupBox;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    Console cons;
    QString sendCons;
    QTextEdit *rPC;
    QFileInfo fichier;

    QCheckBox *ExpTxtQt;
    QTextEdit *ByPQt;
    QTextEdit *PostFixQt;
    QTextEdit *NbMinPtQt;
    QTextEdit *DLRQt;
//    QTextEdit *Pat2Qt;
    QTextEdit *DetectQt;
    QTextEdit *MatchQt;
    QCheckBox *NoMaxQt;
    QCheckBox *NoMinQt;
    QCheckBox *NoUnknownQt;
    QTextEdit *RatioQt;
    QTextEdit *ImInitQtQt;


    std::string  ExpTxtVar;
    std::string  ByPVar;
    std::string  PostFixVar;
    std::string  NbMinPtVar;
    std::string  DLRVar;
    std::string  DetectVar;
    std::string  MatchVar;
    std::string  NoMaxVar;
    std::string  NoMinVar;
    std::string  NoUnknownVar;
    std::string  RatioVar;
    std::string  ImInitVar;


    QTextEdit *Qtr ;
    QRadioButton *mulScale;
    QRadioButton *all;
    QRadioButton *line;
    QRadioButton *file;
    QRadioButton *graph;
    QRadioButton *georeph;
    QMessageBox msgBox;
    QTextEdit *qtt ;
    std::string mode;
    std::string images;
    QTextEdit *regularE;
    QVBoxLayout *boxLayoutV ;
    QVBoxLayout *boxLayoutVOp ;
    QLabel *ql;
    QLabel *labelSizeWm;
    QLabel *labelSizeWM;
    QLabel *labelOptions;
    QTextEdit *resom;
    QTextEdit *resoM;
    std::string resom_str;
    std::string resoM_str;
    QCheckBox* boxTxt;
    std::string txt;

    std::string cmd;

    std::string path_s;
    QString p_stdout;
    QProcess p;
    std::string recup;


};

#endif // TAPIOCA_H
