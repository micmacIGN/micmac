

#ifndef HEADER_MAINWINDOW

#define HEADER_MAINWINDOW


#include <QWidget>
#include <QWindow>
#include <QMenuBar>
#include <QLineEdit>
#include <QLayout>
#include <algorithm>
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
#include <QObject>
#include <QMainWindow>
#include <QDirModel>
#include <QDockWidget>
#include <QPushButton>
#include <QSplitter>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QAction>
#include <QCheckBox>
#include <QGroupBox>
#include <QRadioButton>
#include <QFileInfo>
#include <QRegExp>
#include <QInputDialog>
#include <QTableView>
#include <QStandardItem>
#include <ctime>
#include "StdAfx.h"
#include "Tapioca.h"
#include "Tapas.h"
#include "Campari.h"
#include "ChSys.h"
#include "GrShade.h"
#include "MM2DPosSism.h"
#include "Malt.h"
#include "SaisiePts.h"
#include "Bascule.h"
#include <unistd.h>
#include "Sel.h"
#include "Aperi.h"
#include "Meshlab.h"
#include "CropRPC.h"
#include "Cmd.h"
#include "Tarama.h"
#include "SaisieMasq.h"
#include <iostream>
#include <fstream>
#include "C3dc.h"
#include "Convert2GenBundle.h"
#include "StdAfx.h"
#include <QXmlStreamReader>




class MainWindow : public QMainWindow

{
    Q_OBJECT

public:
    MainWindow();
    void createDock();
    void refreshInterface();
    bool dockE;
    QList<QString> listFileFoldersProject;
    QList<QString> listFileFoldersProject_all;
    QList<QString> checkedItems;
    QList<QString> nameresidu;
    QList<QString> nameresidu2;
    QListWidget itemChecked;
    QList<QListWidgetItem*> a2List;
    QList<QTreeWidgetItem*> a3List;
    QList<QString> qListImg;
    QList<QString> pathsWritten;
    QList<QString> pathsActions;





public slots:
    void loadProject();
    void tapioca();
    void listFileFoldersProject_update();
    void tapas();
    void bascule();
    void campari();
    void malt();
    void saisiePts();
    void dockUpdate();
    void sel();
    void aperi();
    void meshlab();
    void meshlab2(QString);
    void cmd();
    void c3dc();
    void convert();
    void tarama();
    void saisieMasq();
    void to8Bits();
    void grShade();
    void mm2dPosSim();
    void surbri();
    void vino();
    void vino2(QString);
    void centralUpdate();
    void ajoutf();
    void ajoutF();
    void expor();
    void chsys();
    void confirm();
    void imToJpg(std::string);
    void setTextFileReader();
    void vino2click();
    void doSomething(QListWidgetItem*);
    void doSomethingDock(QModelIndex);
    void doSomething2(QListWidgetItem*);
    void findResidu();
    void writeResiduOri(std::string);
    void croprpc();
    std::string readResiduOri();
    void writePaths(QString);
    QList<QString> readPaths();
    void loadExistingProject1();
    void loadExistingProject2();
    void loadExistingProject3();
    void loadExistingProject4();
    void loadExistingProject5();
    void showContextMenu(const QPoint&);




private:
    void createMenu();
    void createCentralWidget();
    void createToolBar();
    void apercu();
    bool isChecked() const;
    void setChecked(bool set);

    int numColumn;
    int outputProc1int;

    std::list<string> aList;
    std::list<string> aListC;

    std::string file;
    std::string imgIcone;
    std::string procSort;
    std::string procproc;
    std::string str;
    string killProc;

    QString GI_MicMacDir;
    QString pathProject;
    QString otherPath;
    QString path;
    QString word;
    QString outputProc1;
    QString output;

    QWidget *QQ;
    QWidget *zoneCentrale;
    QWidget *zoneCentraleBis;
    QWidget *ongletConsole;
    QWidget *ongletFileReader;
    QWidget *ongletVISU;
    QWidget *program_start;
    QWidget *widget;

    QTreeView *treeViewFloders;
    QTableView *treeViewImages;

    QTabWidget *QTabDock;
    QTabWidget *onglets;
    QTabWidget *onglets1 ;
    QTabWidget *onglets2 ;

    QTreeWidgetItem *WidgetItem;

    QVBoxLayout *dockLayoutv;
    QVBoxLayout *layout;

    QVBoxLayout *vboxx;

    QDockWidget *dock;


    QStandardItemModel *mod ;

    QListWidgetItem *item;
    QListWidgetItem *item1;
    QListWidgetItem *item2;
    QListWidgetItem *item3;
    QListWidgetItem *item4;

    QDirModel *model;

    QAbstractItemModel *modelimg;

    QProcess p;
    QProcess p1;
    QProcess Proc1;
    QProcess Proc;
    QProcess Proc2;
    QProcess Proc3;

    QListWidget *ongletPhotos;
    QListWidget *QListImg;

    QTextEdit *lab;
    QTextEdit *lab2;
    QTextEdit *Edit;

    QModelIndex parentIndex;

    QMessageBox msgBox;

    cListOfName aLON;
    cListOfName aLONC;

    QWindow *window;



};


#endif
