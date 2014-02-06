#ifndef SETTINGS_H
#define SETTINGS_H

#include "ui_Settings.h"

#include <QDialog>
#include <QSettings>
#include <iostream>

class cParameters
{
public:
    cParameters();
    ~cParameters(){}

    void setNbFen(QPoint const &aNbFen) { _nbFen = aNbFen;  }
    void setSzFen(QSize aSzFen)         { _szFen = aSzFen;  }
    void setFullScreen(bool fullscreen) { _openFullScreen = fullscreen; }
    void setZoomWindowValue(float aZoom){ _zoomWindow = aZoom; }
    void setDefPtName(QString name)     { _ptName = name;   }
    void setPosition(QPoint pos)        { _position = pos;  }

    QPoint  getNbFen()                  { return _nbFen;    }
    QSize   getSzFen()                  { return _szFen;    }
    bool    getFullScreen()             { return _openFullScreen;  }
    float   getZoomWindowValue()        { return _zoomWindow; }
    QString getDefPtName()              { return _ptName;   }
    QPoint  getPosition()               { return _position; }

    //! Copy operator
    cParameters& operator =(const cParameters& params);

    void    read();
    void    write();

private:
    //main window parameters
    bool        _openFullScreen;
    QPoint      _position;
    QPoint      _nbFen;
    QSize       _szFen;


    //other parameters
    float       _zoomWindow;
    QString     _ptName;
};

//! Dialog to setup display settings
class cSettingsDlg : public QDialog, public Ui::settingsDialog
{
    Q_OBJECT

public:

    //! Default constructor
    cSettingsDlg(QWidget* parent, cParameters *params);
    ~cSettingsDlg();

    void setParameters(cParameters &params);

signals:
    void hasChanged(bool closeWidgets);

protected slots:

    void on_FullscreenCheckBox_clicked();

    void on_okButton_clicked();
    void on_applyButton_clicked();
    void on_resetButton_clicked();
    void on_cancelButton_clicked();

    void on_NBF_x_spinBox_valueChanged(int);
    void on_NBF_y_spinBox_valueChanged(int);

    void on_WindowWidth_spinBox_valueChanged(int);
    void on_WindowHeight_spinBox_valueChanged(int);


protected:

    //! Refreshes dialog to reflect new settings values
    void refresh();

    //! settings
    cParameters *_parameters;

    //! settings copy
    cParameters  _oldParameters;
};


#endif
