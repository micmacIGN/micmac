#ifndef SETTINGS_H
#define SETTINGS_H

#include "ui_Settings.h"

#include <QDialog>
#include <QSettings>

#include "StdAfx.h"

using namespace NS_SaisiePts;

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM = 50.f;
const float GL_MIN_ZOOM = 0.01f;

float zoomClip(float val);

class cParameters
{
public:
    cParameters();
    ~cParameters(){}

    //! Setters
    void setFullScreen(bool aBool)      { _fullScreen = aBool;   }
    void setPosition(QPoint pos)        { _position = pos;       }
    void setNbFen(QPoint const &aNbFen) { _nbFen = aNbFen;       }
    void setSzFen(QSize aSzFen)         { _szFen = aSzFen;       }

    void setZoomWindowValue(int aZoom)  { _zoomWindow = aZoom;   }
    void setDefPtName(QString name)     { _ptName  = name;       }
    void setPostFix(QString name)       { _postFix = name;       }

    void setLineThickness(float val)    { _lineThickness = val;  }
    void setPointDiameter(float val)    { _pointDiameter = val;  }
    void setGamma(float val)            { _gamma = val;          }

    void setSelectionRadius(int val)    { _radius = val;         }

    void setPtCreationMode(eTypePts mode){ _eType = mode;        }
    void setPtCreationWindowSize(double sz){ _sz = sz;           }

    //! Getters
    bool    getFullScreen()             { return _fullScreen;    }
    QPoint  getPosition()               { return _position;      }
    QPoint  getNbFen()                  { return _nbFen;         }
    QSize   getSzFen()                  { return _szFen;         }

    float   getZoomWindowValue()        { return _zoomWindow;    }
    QString getDefPtName()              { return _ptName;        }
    QString getPostFix()                { return _postFix;       }

    float getLineThickness()            { return _lineThickness; }
    float getPointDiameter()            { return _pointDiameter; }
    float getGamma()                    { return _gamma;         }

    int   getSelectionRadius()          { return _radius;        }

    eTypePts getPtCreationMode()        { return _eType;         }
    double getPtCreationWindowSize()    { return _sz;            }

    //! Copy operator
    cParameters& operator =(const cParameters& params);

    //! Comparison operator
    bool operator != (cParameters &p);

    void    read();
    void    write();

private:
    //! Main window parameters
    bool        _fullScreen;
    QPoint      _position;
    QPoint      _nbFen;
    QSize       _szFen;

    //! Drawing settings
    float       _lineThickness;
    float       _pointDiameter;
    float       _gamma;

    //! Other parameters
    float       _zoomWindow;
    QString     _ptName;
    QString     _postFix;
    int         _radius;

    //! Point creation mode
    eTypePts    _eType;
    double      _sz;
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

    void enableMarginSpinBox(bool show = true);

signals:
    void nbFenChanged(bool closeWidgets);

    void lineThicknessChanged(float);
    void pointDiameterChanged(float);
    void gammaChanged(float);
    void zoomWindowChanged(float);
    void selectionRadiusChanged(int);
    void prefixTextEdit(QString);

protected slots:

    void on_okButton_clicked();
    void on_applyButton_clicked();
    void on_resetButton_clicked();
    void on_cancelButton_clicked();

    //!layout settings
    void on_NBF_x_spinBox_valueChanged(int);
    void on_NBF_y_spinBox_valueChanged(int);
    void on_WindowWidth_spinBox_valueChanged(int);
    void on_WindowHeight_spinBox_valueChanged(int);

    //!drawing settings
    void on_LineThickness_doubleSpinBox_valueChanged(double);
    void on_PointDiameter_doubleSpinBox_valueChanged(double);
    void on_GammaDoubleSpinBox_valueChanged(double);

    //!other display settings
    void on_zoomWin_spinBox_valueChanged(int);
    void on_RadiusSpinBox_valueChanged(int);
    void on_PrefixTextEdit_textChanged(QString);

    //!point creation mode
    void on_radioButtonStd_toggled(bool);
    void on_radioButtonMin_toggled(bool);
    void on_radioButtonMax_toggled(bool);
    void on_doubleSpinBoxSz_valueChanged(double);


protected:

    //! Refreshes dialog to reflect new settings values
    void refresh();

    //! settings
    cParameters *_parameters;

    //! settings copy
    cParameters  _oldParameters;
};


#endif
