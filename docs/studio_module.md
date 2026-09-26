# histox.studio

This module contains the HistoX Studio visualization tool. See [HistoX Studio: Live Visualization](studio.html#studio) for more information.

*class*histox.studio.Sidebar(*viz: Studio*)[[source]](_modules/histox/studio.html#Sidebar)

Sidebar for Studio, rendering a navigation bar and widgets.

add_widgets(*widgets*)[[source]](_modules/histox/studio.html#Sidebar.add_widgets)

Add widget extension(s).

Add widgets to the navigation sidebar. The `.tag` property is used
as a unique identifier for the widget. The `.icon` property should
be a path to an image file used for rendering the sidebar navigation
icon. `.icon_highlighted` property should be a path to an image file
used for rendering a hovered navigation icon.

The widget should implement `__call__()` and `.close()` methods
for rendering the imgui GUI and cleanup, respectively.

Parameters:

**widgets** (list(`histox.studio.widgets.Widget`)) - List of
widgets to add as extensions. These should be classes, not
instantiated objects.

*property*content_width

Widget width.

draw()[[source]](_modules/histox/studio.html#Sidebar.draw)

Draw the sidebar and render all widgets.

full_button(*text*, *width=None*, ***kwargs*)[[source]](_modules/histox/studio.html#Sidebar.full_button)

Render a button that spans the full width of the sidebar.

The color of the button is determined through the loaded theme,
`bright_button` properties.

Parameters:

- **text** (*str*) - Text of the button.
- **width** (*int**,**optional*) - Width of the button. If not specified,
uses a width that spans the width of the sidebar.

Keyword Arguments:

**enabled** (*bool*) - Whether the button is enabled.

Returns:

Whether the button was clicked.

Return type:

bool

full_button2(*text*, *width=None*, ***kwargs*)[[source]](_modules/histox/studio.html#Sidebar.full_button2)

Render a button that spans the full width of the sidebar.

The color of the button is determined through the loaded theme,
`bright_button2` properties.

Parameters:

- **text** (*str*) - Text of the button.
- **width** (*int**,**optional*) - Width of the button. If not specified,
uses a width that spans the width of the sidebar.

Keyword Arguments:

**enabled** (*bool*) - Whether the button is enabled.

Returns:

Whether the button was clicked.

Return type:

bool

*property*full_width

Width of the expanded sidebar, including navigation and widgets.

large_image_button(*image_name*, *size=None*)[[source]](_modules/histox/studio.html#Sidebar.large_image_button)

Render a small button for the sidebar.

Parameters:

- **image_name** (*str*) - Name of the image to render on the button.
Valid names include 'gear', 'circle_lightning', 'circle_plus',
'pencil', 'folder', 'floppy', 'model_loaded', 'extensions',
'project', 'slide', 'model', and 'heatmap'.
- **size** (*int*) - Simage button. Defaults to 64.

Returns:

If the button was clicked.

Return type:

bool

remove_widget(*tag*)[[source]](_modules/histox/studio.html#Sidebar.remove_widget)

Remove a widget from Studio.

Parameters:

**widget** (`histox.studio.widgets.Widget`) - Widget to remove.
This should be a class, not an instantiated object.

small_button(*image_name*)[[source]](_modules/histox/studio.html#Sidebar.small_button)

Render a small button for the sidebar.

Parameters:

**image_name** (*str*) - Name of the image to render on the button.
Valid names include 'vips', 'cucim', 'lowmem', 'ellipsis',
'gear', and 'refresh'.

Returns:

If the button was clicked.

Return type:

bool

*property*theme

Active Studio theme.

*class*histox.studio.Studio(*low_memory: bool = False*, *widgets: List[Any] | None = None*, *skip_tk_init: bool = False*, *theme: StudioTheme | None = None*)[[source]](_modules/histox/studio.html#Studio)

*property*P

HistoX project currently in use.

add_to_render_pipeline(*renderer: Any*, *name: str | None = None*) → None[[source]](_modules/histox/studio.html#Studio.add_to_render_pipeline)

Add a renderer to the rendering pipeline.

add_widgets(*widgets: Widget*) → None[[source]](_modules/histox/studio.html#Studio.add_widgets)

Add widget extension(s).

Add widgets to Studio and the sidebar. The `.tag` property is used
as a unique identifier for the widget. The `.icon` property should
be a path to an image file used for rendering the sidebar navigation
icon. `.icon_highlighted` property should be a path to an image file
used for rendering a hovered navigation icon.

The widget should implement `__call__()` and `.close()` methods
for rendering the imgui GUI and cleanup, respectively.

Parameters:

**widgets** (list(`histox.studio.widgets.Widget`)) - List of
widgets to add as extensions. These should be classes, not
instantiated objects.

ask_load_heatmap()[[source]](_modules/histox/studio.html#Studio.ask_load_heatmap)

Prompt user for location of exported heatmap (*.npz) and load.

ask_load_model()[[source]](_modules/histox/studio.html#Studio.ask_load_model)

Prompt user for location of a model and load.

ask_load_project()[[source]](_modules/histox/studio.html#Studio.ask_load_project)

Prompt user for location of a project and load.

ask_load_slide()[[source]](_modules/histox/studio.html#Studio.ask_load_slide)

Prompt user for location of a slide and load.

ask_zoom_to_mpp() → None[[source]](_modules/histox/studio.html#Studio.ask_zoom_to_mpp)

Prompt the user to zoom to a specific microns-per-pixel (MPP).

autoload(*path*, *ignore_errors=False*)[[source]](_modules/histox/studio.html#Studio.autoload)

Automatically load a path, detecting the type of object to load.

Supports slides, models, projects, and other items supported by
widgets if the widget has implemented a .drag_and_drop_hook function.

Parameters:

- **path** (*str*) - Path to file to load.
- **ignore_errors** (*bool*) - Gracefully handle errors.

center_next_window(*width*, *height*)[[source]](_modules/histox/studio.html#Studio.center_next_window)

Center the next imgui window.

Parameters:

- **width** (*int*) - Width of the next window.
- **height** (*int*) - Height of the next window.

clear_message(*msg: str | None = None*) → bool[[source]](_modules/histox/studio.html#Studio.clear_message)

Clear a specific message from display, if the message is being shown.

Parameters:

**msg** (*str*) - Message to clear.

Returns:

Whether message was cleared from display.

Return type:

bool

clear_model_results() → None[[source]](_modules/histox/studio.html#Studio.clear_model_results)

Clear all model results and associated images.

clear_overlay() → None[[source]](_modules/histox/studio.html#Studio.clear_overlay)

Remove the current overlay image, include heatmaps and masks.

clear_result() → None[[source]](_modules/histox/studio.html#Studio.clear_result)

Clear all shown results and images.

clear_status_message() → None[[source]](_modules/histox/studio.html#Studio.clear_status_message)

Clear the status message from the status bar.

close() → None[[source]](_modules/histox/studio.html#Studio.close)

Close the application and renderer.

close_model(*now: bool = False*) → None[[source]](_modules/histox/studio.html#Studio.close_model)

Close the currently loaded model.

Parameters:

**now** (*bool*) - Close the model now, instead of at the end of the frame.
Defaults to False (closes model at frame end).

close_slide(*now: bool = False*) → None[[source]](_modules/histox/studio.html#Studio.close_slide)

Close the currently loaded slide.

Parameters:

**now** (*bool*) - Close the slide now, instead of at the end of the frame.
Defaults to False (closes slide at frame end).

collapsing_header(*text*, ***kwargs*)[[source]](_modules/histox/studio.html#Studio.collapsing_header)

Render a collapsing header using the active theme.

Examples

Render a collapsing header that is open by default.

> ```
> if viz.collapsing_header("Header", default=True):
> imgui.text("Text underneath")
> ```

Parameters:

**text** (*str*) - Header text.

collapsing_header2(*text*, ***kwargs*)[[source]](_modules/histox/studio.html#Studio.collapsing_header2)

Render a second-level collapsing header using the active theme.

Examples

Render a collapsing header that is open by default.

> ```
> if viz.collapsing_header("Header", default=True):
> imgui.text("Text underneath")
> ```

Parameters:

**text** (*str*) - Header text.

decrease_tile_zoom() → None[[source]](_modules/histox/studio.html#Studio.decrease_tile_zoom)

Decrease zoom of tile view by half.

defer_rendering(*num_frames: int = 1*) → None[[source]](_modules/histox/studio.html#Studio.defer_rendering)

Defer rendering for a number of frames.

dim_text(*dim=True*)[[source]](_modules/histox/studio.html#Studio.dim_text)

Render dim text.

Examples

Render dim text.

> ```
> with studio.dim_text():
> imgui.text('This is dim')
> ```

draw_frame() → None[[source]](_modules/histox/studio.html#Studio.draw_frame)

Main draw loop.

*static*get_default_widgets() → List[Any][[source]](_modules/histox/studio.html#Studio.get_default_widgets)

Returns a list of the default non-mandatory extension widgets.

get_extension(*tag: str*) → Widget | None[[source]](_modules/histox/studio.html#Studio.get_extension)

Returns a given widget (extension) by tag.

Parameters:

**tag** (*str*) - Tag of the widget to search for.

Returns:

histox.studio.widgets.Widget if found, else None

get_renderer(*name: str | None = None*) → Renderer | None[[source]](_modules/histox/studio.html#Studio.get_renderer)

Check for the given additional renderer in the rendering pipeline.

Parameters:

**name** (*str*) - Name of the renderer to check for. If None,
returns the main renderer.

Returns:

Renderer if name is a recognized renderer, otherwise None

get_widget(*name: str*) → Widget[[source]](_modules/histox/studio.html#Studio.get_widget)

Returns a given widget by class name.

Parameters:

**name** (*str*) - Name of the widget to search for.

Returns:

histox.studio.widgets.Widget

Raises:

**ValueError** - If the widget could not be found.

has_live_viewer() → bool[[source]](_modules/histox/studio.html#Studio.has_live_viewer)

Check if the current viewer is a live viewer (e.g. camera feed).

has_uq() → bool[[source]](_modules/histox/studio.html#Studio.has_uq)

Check if the current model supports uncertainty quantification.

header(*text*)[[source]](_modules/histox/studio.html#Studio.header)

Render a header using the active theme.

Parameters:

**text** (*str*) - Text for the header. Text will be rendered in
uppercase.

header_with_buttons(*text*)[[source]](_modules/histox/studio.html#Studio.header_with_buttons)

Render a widget header with ability to add buttons.

Examples

Render a header with a gear icon.

> ```
> with studio.header_with_buttons('Button'):
> # Right align the button
> x_width = imgui.get_content_region_max()[0]
> imgui.same_line(x_width - 30)
> cx, cy = imgui.get_cursor_pos()
> imgui.set_cursor_position((cx, cy-5))
> 
> # Render the button
> if sidebar.small_button('gear'):
> do_something()
> ```

Parameters:

**text** (*str*) - Text for the header. Text will be rendered in
uppercase.

highlighted(*enable: bool = True*)[[source]](_modules/histox/studio.html#Studio.highlighted)

Render highlighted text.

Parameters:

**enable** (*bool*) - Whether to enable highlighting.

Examples

Render highlighted text.

> ```
> with studio.highlighted(True):
> imgui.text('This is highlighted')
> ```

increase_tile_zoom() → None[[source]](_modules/histox/studio.html#Studio.increase_tile_zoom)

Increase zoom of tile view two-fold.

is_mouse_down(*mouse_idx: int = 0*) → bool[[source]](_modules/histox/studio.html#Studio.is_mouse_down)

Check if the mouse is currently down.

is_mouse_released(*mouse_idx: int = 0*) → bool[[source]](_modules/histox/studio.html#Studio.is_mouse_released)

Check if the mouse was released.

load_heatmap(*path: str | [Heatmap](heatmap.html#histox.Heatmap)*) → None[[source]](_modules/histox/studio.html#Studio.load_heatmap)

Load a saved heatmap (*.npz).

Parameters:

**path** (*str*) - Path to exported heatmap in *.npz format, as generated
by Heatmap.save() or Heatmap.save_npz().

load_model(*model: str*, *ignore_errors: bool = False*) → None[[source]](_modules/histox/studio.html#Studio.load_model)

Load the given model.

Parameters:

- **model** (*str*) - Path to HistoX model (in either backend).
- **ignore_errors** (*bool*) - Do not fail if an error is encountered.
Defaults to False.

load_project(*project: str*, *ignore_errors: bool = False*) → None[[source]](_modules/histox/studio.html#Studio.load_project)

Load the given project.

Parameters:

- **project** (*str*) - Path to HistoX project.
- **ignore_errors** (*bool*) - Do not fail if an error is encountered.
Defaults to False.

load_slide(*slide: str*, ***kwargs*) → None[[source]](_modules/histox/studio.html#Studio.load_slide)

Load the given slide.

Parameters:

- **slide** (*str*) - Path to whole-slide image.
- **stride** (*int**,**optional*) - Stride for tiles. 1 is non-overlapping
tiles, 2 is tiles with 50% overlap, etc. Defaults to 1.
- **ignore_errors** (*bool*) - Do not fail if an error is encountered.
Defaults to False.

*property*model

Tensorflow/PyTorch model currently in use.

mouse_input_is_suspended() → bool[[source]](_modules/histox/studio.html#Studio.mouse_input_is_suspended)

Check if mouse input handling is suspended.

*property*mouse_is_over_viewer

Mouse is currently over the main viewer.

*property*offset_x

Main window offset (x), in points.

*property*offset_x_pixels

Main window offset (x), in pixels.

*property*offset_y

Main window offset (y), in points.

*property*offset_y_pixels

Main window offset (y), in pixels.

print_error(*error: str*) → None[[source]](_modules/histox/studio.html#Studio.print_error)

Print the given error message.

reload_model() → None[[source]](_modules/histox/studio.html#Studio.reload_model)

Reload the current model.

reload_viewer() → None[[source]](_modules/histox/studio.html#Studio.reload_viewer)

Reload the current main viewer.

reload_wsi(*slide: str | [WSI](slide.html#histox.slide.WSI) | None = None*, *stride: int | None = None*, *use_rois: bool = True*, *tile_px: int | None = None*, *tile_um: str | int | None = None*, ***kwargs*) → bool[[source]](_modules/histox/studio.html#Studio.reload_wsi)

Reload the currently loaded Whole-Slide Image.

Parameters:

- **path** (*str**or**hx.WSI**,**optional*) - Slide to reload. May be a path
or a hx.WSI object. If not provided, will reload the
currently loaded slide.
- **stride** (*int**,**optional*) - Stride to use for the loaded slide. If not
provided, will use the stride value from the currently loaded
slide.
- **use_rois** (*bool*) - Use ROIs from the loaded project, if available.

Returns:

True if slide loaded successfully, False otherwise.

Return type:

bool

remove_from_render_pipeline(*name: str*)[[source]](_modules/histox/studio.html#Studio.remove_from_render_pipeline)

Remove a renderer from the render pipeline.

Remove a renderer added with `.add_to_render_pipeline()`.

Parameters:

**name** (*str*) - Name of the renderer to remove.

remove_widget(*widget: Widget*) → None[[source]](_modules/histox/studio.html#Studio.remove_widget)

Remove a widget from Studio.

Parameters:

**widget** (`histox.studio.widgets.Widget`) - Widget to remove.
This should be a class, not an instantiated object.

reset_background()[[source]](_modules/histox/studio.html#Studio.reset_background)

Reset the Studio background to the default theme color.

reset_tile_zoom() → None[[source]](_modules/histox/studio.html#Studio.reset_tile_zoom)

Reset tile zoom level.

resume_keyboard_input() → bool[[source]](_modules/histox/studio.html#Studio.resume_keyboard_input)

Resume keyboard input handling.

resume_mouse_input_handling()[[source]](_modules/histox/studio.html#Studio.resume_mouse_input_handling)

Resume mouse input handling.

set_grid_overlay(*grid: ndarray*, ***, *tile_um: int | None = None*, *stride_div: int | None = None*, *mpp: float | None = None*, *original: ndarray | None = None*) → None[[source]](_modules/histox/studio.html#Studio.set_grid_overlay)

Set the grid overlay to the given grid.

Parameters:

**grid** (*np.ndarray*) - Grid to render as an overlay.

Keyword Arguments:

- **tile_um** (*int**,**optional*) - Tile size, in microns. If None, uses
the tile size of the currently loaded slide.
- **stride_div** (*int**,**optional*) - Stride divisor. If None, uses
the stride divisor of the currently loaded slide.
- **mpp** (*float**,**optional*) - Microns per pixel. If None, uses
the MPP of the currently loaded slide.
- **original** (*np.ndarray**,**optional*) - Original grid values before any
colorization or other modifications. Used for displaying the
tooltip when alt-hovering. Defaults to None.

set_message(*msg: str*) → None[[source]](_modules/histox/studio.html#Studio.set_message)

Set a message for display.

set_overlay(*overlay: ndarray*, *method: int*, ***, *original: ndarray | None = None*) → None[[source]](_modules/histox/studio.html#Studio.set_overlay)

Configure the overlay to be applied to the current view screen.

Overlay is a numpy array, and method is a flag indicating the
method to use when showing the overlay.

If `method` is `hx.studio.OVERLAY_WSI`, the array will be mapped
to the entire whole-slide image, without offsets.

If `method` is `hx.studio.OVERLAY_GRID`, the array is interpreted
as having been generated from the slide's grid, meaning that an offset
will be applied to ensure that the overlay is aligned properly.

If `method` is `hx.studio.OVERLAY_VIEW`, the array is interpreted
as an overlay that is applied only to the area of the slide
currently in view.

Parameters:

- **overlay** (*np.ndarray*) - Overlay to render.
- **method** (*int*) - Mapping method for linking the overlay to the
whole-slide image.

Keyword Arguments:

**original** (*np.ndarray**,**optional*) - Original grid values before any
colorization or other modifications. Used for displaying the
tooltip when alt-hovering. Defaults to None.

set_prediction_message(*msg: str*) → None[[source]](_modules/histox/studio.html#Studio.set_prediction_message)

Set the prediction message to display under the tile outline.

set_status_message(*message: str*, *description: str | None = None*, ***, *color: Tuple[float, float, float] | None = (0.7, 0, 0, 1)*, *text_color: Tuple[float, float, float, float] = (1, 1, 1, 1)*, *rounding: int = 0*) → None[[source]](_modules/histox/studio.html#Studio.set_status_message)

Set the status message to display in the status bar.

set_viewer(*viewer: Any*) → None[[source]](_modules/histox/studio.html#Studio.set_viewer)

Set the main viewer.

Parameters:

**viewer** (`histox.studio.gui.viewer.Viewer`) - Viewer to use.

*property*show_overlay

An overlay (e.g. tile filter or heatmap) is currently being shown
over the main view.

suspend_keyboard_input() → bool[[source]](_modules/histox/studio.html#Studio.suspend_keyboard_input)

Suspend keyboard input handling.

suspend_mouse_input_handling()[[source]](_modules/histox/studio.html#Studio.suspend_mouse_input_handling)

Suspend mouse input handling.

*property*tile_preview_enabled

Show a tile preview when right clicking.