#ifndef __CONTEXTMENUMANAGER__
#define __CONTEXTMENUMANAGER__

#include <QMenu>
#include <QSignalMapper>
#include <QIcon>
#include <QInputDialog>
#include <QDialogButtonBox>

#include <QWidget>
#include <QAction>

#include "3DObject.h"

class ContextMenu : public QWidget
{
    Q_OBJECT

public:
    ContextMenu(){}

    void createContexMenuActions();

    void setPolygon(cPolygon * poly){ _polygon = poly; }
    void setPos(QPointF pt) { _lastPosImage = pt; }

    cPolygon    *_polygon;

    QAction     *_showNames;
    QAction     *_showRefuted;
    QAction     *_rename;

    QAction     *_AllW;
    QAction     *_ThisP;
    QAction     *_ThisW;

    QAction     *_validate;
    QAction     *_dubious;
    QAction     *_refuted;
    QAction     *_noSaisie;
    QAction     *_highLight;

public slots:

    void setPointState(int state);

    void highlight();

    void rename();

    void showNames();

    void showRefuted();

private :

    QPointF                 _lastPosImage; //copy of QGLWidget's m_lastPosImage

    QSignalMapper*          _signalMapper;
};

#endif
