#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "general/CMake_defines.h"

#if ELISE_QT

#include "StdAfx.h"
#include "saisieQT_window.h"
#include "general/visual_buttons.h"


using namespace std;

enum eInputType
{
    eIT_LineEdit,
    eIT_ComboBox,
    eIT_SpinBox,
    eIT_DoubleSpinBox,
    eIT_None
};

class cInputs
{
public:
    cInputs(cMMSpecArg, vector < pair < int, QWidget* > >);

    bool        IsOpt()     { return mArg.IsOpt(); }
    cMMSpecArg  Arg()       { return mArg;    }
    int         Type();
    vector < pair < int, QWidget*> >    Widgets()    { return vWidgets; }
    int         NbWidgets() { return (int)vWidgets.size(); }

private:
    cMMSpecArg  mArg;
    vector < pair < int, QWidget* > >   vWidgets;
};

class visual_MainWindow : public QWidget
{
    Q_OBJECT

public:
    visual_MainWindow(vector<cMMSpecArg> &aVAM,
                      vector<cMMSpecArg> &aVAO,
                      std::string aFirstArg = "",
                      QString aLastDir = QDir::currentPath(),
                      QWidget *parent = 0
                      );

    ~visual_MainWindow();

    void moveArgs(vector<cMMSpecArg> &aVAM, vector<cMMSpecArg> &aVAO);

    void add_label     (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_combo     (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_select    (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_1i_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_2i_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_3i_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_4i_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_1d_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_2d_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_3d_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_4d_SpinBox(QGridLayout*, QWidget*, int, cMMSpecArg);

    void set_argv_recup(string);

    void buildUI(const vector<cMMSpecArg>& aVA, QGridLayout* layout, QWidget* parent);

    void addGridLayout(const vector<cMMSpecArg>& aVA, QString pageName, bool addSpace =true);

    bool getSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar ="");
    bool getDoubleSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar ="");

    QDoubleSpinBox* create_1d_SpinBox(QGridLayout*, QWidget*, int, int);
    QSpinBox*       create_1i_SpinBox(QGridLayout*, QWidget*, int, int);

    void add_saisieButton(QGridLayout *layout, int aK, bool normalize);

    void saveSettings();

    void setLastDir(QString filename);

public slots:

    void onRunCommandPressed();
    void onSelectFilePressed(int);
    void onSelectFileRPPressed(int);
    void onSelectImgsPressed(int);
    void onSelectDirPressed(int);
    void onSaisieButtonPressed(int, bool normalize);
    void _adjustSize(int);

    void onRectanglePositionChanged(QVector <QPointF> pointsTerrain);
    void onSaisieQtWindowClosed();

    void setShowDialog(int state);

signals:

    void newX0Position(int);
    void newY0Position(int);
    void newX1Position(int);
    void newY1Position(int);

    void newX0Position(double);
    void newY0Position(double);
    void newX1Position(double);
    void newY1Position(double);

protected:

    void resizeEvent(QResizeEvent *);
    void closeEvent(QCloseEvent *event);
    void keyPressEvent(QKeyEvent* event);

    void checkGeoref(QString aNameFile);

    cFileOriMnt  _m_FileOriMnt;

    string       argv_recup;

    QToolBox*    toolBox;
    QPushButton* runCommandButton;
    QCheckBox*   showPromptDialog;

    vector <QLineEdit*> vLineEdit;    //LineEdit: display what has been selected (images, files, directories)

    vector <cInputs*>   vInputs;

    QString             mlastDir;

    string              mFirstArg;    //truc&astuces: stores the first arg (for Tapioca)

    SaisieQtWindow*     _SaisieWin;

    int                 _curIdx;

    bool                _showDialog; //do we show "Job done" at the end of process

    bool                _bMaltGeomImg;
    const QVector<QPointF> transfoTerrain(QVector<QPointF> res);
};

list<string> listPossibleValues(const cMMSpecArg & anArg);
void ShowEnum(const cMMSpecArg & anArg);
void showErrorMsg(QApplication &app, std::vector <std::string> vStr);

#endif // ELISE_QT

#endif // MAINWINDOW_H
