#ifndef CROPRPC
#define CROPRPC


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
#include "CropRPC.h"
#include "Console.h"


class CropRPC : public QWidget
{

    Q_OBJECT


public:
    CropRPC(QList <QString>, QList <QString>, QString);
    QList<QString> listImages;
    QMessageBox msgBox2;
    QString msg1();
    QDialog qDi;




signals:
    void signal_a();

public slots:
    void mm3d();
    void stopped();
    void oriSearch();
    void oriSearch1();
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
    QLabel *oriFileCrop;
    QLabel *directoryOutput;
    QLabel *labelOptions;
    QTextEdit *oriFileCropTxt;
    QTextEdit *directoryOutputTxt;
    QTextEdit *patternTxt;
    std::string oriFileCropTxt_str;
    std::string directoryOutputTxt_str;
    std::string patternTxt_str;
    QCheckBox* boxTxt;
    std::string txt;
    std::string cmd;
    std::string path_s;
    QString p_stdout;
    QProcess p;
    std::string recup;
    QString aPath;

    QTextEdit *SzQt;
    QTextEdit *SzQt2;
    QTextEdit *OrgQt;
    QTextEdit *OrgQt2;

    std::string  SzVar;
    std::string  OrgVar;


};
#endif // CROPRPC

