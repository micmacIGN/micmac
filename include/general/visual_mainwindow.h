#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#if(ELISE_QT5)



#include "StdAfx.h"

#ifdef Int
    #undef Int
#endif

#include <QMainWindow>

#include <QGridLayout>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QToolBox>
#include <QFileDialog>
#include <QDesktopWidget>
#include <QApplication>

using namespace std;

enum eInputType
{
    eLineEdit,
    eComboBox,
    eInteger
};

class visual_MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    visual_MainWindow(vector<cMMSpecArg> & aVAM, vector<cMMSpecArg> & aVAO, QWidget *parent = 0);
    ~visual_MainWindow();

    void create_combo(QGridLayout *layout, QWidget *parent, int, list<string>);
    void create_comment(QGridLayout *layout, QWidget *parent, string, int ak);
    void create_select(QGridLayout* layout, QWidget* parent, int, cMMSpecArg);
    void create_champ_int(QGridLayout* layout, QWidget* parent, int);
    void set_argv_recup(string);

    void buildUI(vector<cMMSpecArg>& aVAM, QGridLayout* layout, QWidget* parent, bool isOpt=false);

public slots:

    void onRunCommandPressed();
    void onSelectFilePressed(int);
    void onSelectImgsPressed(int);
    void onSelectDirPressed(int);
    void _adjustSize(int);

protected:

    void resizeEvent(QResizeEvent *);

    int id_unique;
    string argv_recup;

    QWidget*     mainWidget;
    QGridLayout* gridLayout;

    QLabel*      label;
    QComboBox*   Combo;
    QLineEdit*   select_LineEdit;
    QPushButton* select_Button;
    QSpinBox*    SpinBox;
    QToolBox*    toolBox;
    QPushButton* runCommandButton;

    vector <QComboBox*> vEnumValues;    //enum values
    vector <QLineEdit*> vLineEdit;      //LineEdit: display what has been selected (images, files, directories)
    //vector <QLabel*>    vComments;      //comments

    vector <eInputType> vInputTypes;
    vector <QWidget*>   vInputs;

    QString        mlastDir;

    list <string>  bList;
};

list<string> listPossibleValues(const cMMSpecArg & anArg);
void ShowEnum(const cMMSpecArg & anArg);

#endif //ELISE_QT5

#endif // MAINWINDOW_H
