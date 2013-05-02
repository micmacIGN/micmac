#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <QtOpenGL/QGLWidget>
#include <QGLContext>
#include <QUrl>

#include "Cloud.h"
#include "util.h"
#include "mmGuiParameters.h"
#include "mmVector3.h"

//! Model view matrix size (OpenGL)
static const unsigned OPENGL_MATRIX_SIZE = 16;

class ViewportParameters// : public ccSerializableObject
{
public:
    //! Default constructor
    ViewportParameters();

    //! Copy constructor
    ViewportParameters(const ViewportParameters& params);

    //! Current pixel size (in 'current unit'/pixel)
    /** This scale is valid eveywhere in ortho. mode
        or at the focal distance in perspective mode.
        Warning: doesn't take current zoom into account!
    **/
    float pixelSize;

    //! Current zoom
    float zoom;

    //! Visualization matrix (rotation only)
    //ccGLMatrix viewMat;

    //! Point size
    float defaultPointSize;
    //! Line width
    float defaultLineWidth;

    //! Whether view is centered on displayed scene (true) or on the user eye (false)
    /** Always true for ortho. mode.
    **/
    bool objectCenteredView;

    //! Rotation pivot point (for object-centered view modes)
    Vector3 pivotPoint;

    //! Camera center (for perspective mode)
    Vector3 cameraCenter;

    //! Camera F.O.V. (field of view - for perspective mode only)
    float fov;
    //! Camera aspect ratio (for perspective mode only)
    float aspectRatio;
};

class GLWidget : public QGLWidget {
private:
    QVector <Cloud_::Cloud> m_ply;

    Q_OBJECT // must include this if you use Qt signals/slots

public:
    GLWidget(QWidget *parent = NULL);

    void addPly( const QString & );

    //! Interaction mode (with the mouse!)
    enum INTERACTION_MODE { TRANSFORM_CAMERA,
                            TRANSFORM_ENTITY,
                            SEGMENT_POINTS
    };

    //! Default message positions on screen
    enum MessagePosition {  LOWER_LEFT_MESSAGE,
                            UPPER_CENTER_MESSAGE,
                            SCREEN_CENTER_MESSAGE
    };

    //! Message type
    enum MessageType {  CUSTOM_MESSAGE,
                        SCREEN_SIZE_MESSAGE,
                        SUN_LIGHT_STATE_MESSAGE,
                        CUSTOM_LIGHT_STATE_MESSAGE,
                        MANUAL_TRANSFORMATION_MESSAGE,
                        MANUAL_SEGMENTATION_MESSAGE
    };

    //! Displays a status message in the bottom-left corner
    /** WARNING: currently, 'append' is not supported for SCREEN_CENTER_MESSAGE
        \param message message (if message is empty and append is 'false', all messages will be cleared)
        \param pos message position on screen
        \param type message type (if not custom, only one message of this type at a time is accepted)
    **/
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos,
                                   MessageType type=CUSTOM_MESSAGE);

    bool hasCloudLoaded(){return m_bCloudLoaded;}

    void setCloudLoaded(bool isLoaded) { m_bCloudLoaded = isLoaded; }

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(MM_VIEW_ORIENTATION orientation, bool redraw=true);

    //! Invalidate current visualization state
    /** Forces view matrix update and 3D/FBO display.
    **/
    void invalidateVisualization();

    void invalidateViewport();

    //! Returns the current (OpenGL) view matrix as a double array
    const double* getModelViewMatd();

    //! Updates current zoom
    void updateZoom(float zoomFactor);

    //! Sets current zoom
    void setZoom(float value);

    //! Returns the current (OpenGL) projection matrix as a double array
    const double* getProjectionMatd();


public slots:
    void zoomGlobal();
    void redraw();

    //called when recieving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    //! Signal emitted during 3D pass of OpenGL display process
    /** Any object connected to this slot can draw additional stuff in 3D.
    **/
    void drawing3D();

    //! Signal emitted when the mouse wheel is rotated
    void mouseWheelRotated(float wheelDelta_deg);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    //void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);

    //! Initialization state
    bool m_initialized;

    //inherited from QWidget (drag & drop support)
    virtual void dragEnterEvent(QDragEnterEvent* event);
    virtual void dropEvent(QDropEvent* event);

    void getContext(glDrawContext& context);

    void draw3D(bool doDrawCross);

    void drawGradientBackground();

    //Projections controls
    void recalcModelViewMatrix();
    void recalcProjectionMatrix();
    void setStandardOrthoCenter();

    void drawCross();

    //! GL context width
    int m_glWidth;
    //! GL context height
    int m_glHeight;

    //! Sun light position
    /** Relative to screen.
    **/
    float m_sunLightPos[4];

    void glEnableSunLight();

    //! Returns current font size
    virtual int getFontPointSize() const;
    //! Sets current font size
    virtual void setFontPointSize(int pixelSize);

    //! Default font
    QFont m_font;

    //! States if a cloud is already loaded
    bool m_bCloudLoaded;

    //! Complete visualization matrix (GL style - double version)
    double m_viewMatd[OPENGL_MATRIX_SIZE];

    //! Whether the projection matrix is valid (or need to be recomputed)
    bool m_validProjectionMatrix;

    //! Whether the model veiw matrix is valid (or need to be recomputed)
    bool m_validModelviewMatrix;

    //! Projection matrix (GL style - double version)
    double m_projMatd[OPENGL_MATRIX_SIZE];

    //! Whether FBO should be updated (or simply displayed as a texture = faster!)
    bool m_updateFBO;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

    //! Temporary Message to display in the lower-left corner
    struct MessageToDisplay
    {
        //! Message
        QString message;
        //! Message end time (sec)
        int messageValidity_sec;
        //! Message position on screen
        MessagePosition position;
        //! Message type
        MessageType type;
    };

    //! List of messages to display
    std::list<MessageToDisplay> m_messagesToDisplay;

    //! Point list for polygonal selection
    QVector < QPoint > m_polygon;

    //! Fit view to point cloud
    bool m_bFitCloud;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;

};




#endif  /* _GLWIDGET_H */

