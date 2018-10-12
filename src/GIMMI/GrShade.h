#ifndef GrShade_H
#define GrShade_H



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
#include "GrShade.h"
#include "Console.h"

class GrShade : public QWidget
{

    Q_OBJECT


public:
    GrShade(QString);
    QList<QString> listImages;
    QMessageBox msgBox2;
    QDialog qDi;
    QString msg1();




signals:
    void signal_a();

public slots:
    void msg();
    void readyReadStandardOutput();
    void stopped();

private:
    QButtonGroup *modeGroupBox;
    QCheckBox* imagesCheckBox;
    QList<QCheckBox*> listeCasesImages;
    Console cons;
    QString sendCons;
    QTextEdit *rPC;
    QFileInfo fichier;
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





#endif // GRSHADE

