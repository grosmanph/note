from utils.gaussian_profiles import LG_transverse_complex_amplitude
from utils.gaussian_profiles import HG_transverse_complex_amplitude
from utils.phase_screen import generate_phase_screen
from utils.bessel import apply_inverse_bessel_j1_2d
from utils.functions import halve_element, fourier_rescale, upscale_image_nearest

from PIL import Image
from datetime import datetime
import numpy as np
import json


def complex_field_amplitude(w0: float, k: float, mode_indexes: list, z: float, xrange: np.array, yrange: np.array, mode_name: str, divergence: bool) -> np.array:
    # create a dictionary to switch transverse modes
    switch_mode = {
        "HG": HG_transverse_complex_amplitude,
        "LG": LG_transverse_complex_amplitude
    }

    # calculate transverse meshgrid
    x, y = np.meshgrid(xrange, yrange)

    if mode_name == "LG":
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.angle(x + 1j * y)
        x, y = (r, theta)

    # calculate the transverse components of the complex field
    if mode_name not in switch_mode:
        raise ValueError(f"Selected mode {mode_name} not supported")

    i1, i2 = mode_indexes
    e_complex = switch_mode[mode_name](x, y, z, k, w0, i1, i2, divergence)

    return e_complex


def calculate_hologram(e_complex: np.array, xy_grid_freq: list, xy_pixels: list, xrange: np.array, yrange: np.array) -> np.array:
    # Normalize the complex amplitude
    e_complex = e_complex / np.sqrt(np.sum(np.abs(e_complex) ** 2))
    phase = np.angle(e_complex)
    amplitude = np.abs(e_complex)

    v = amplitude / amplitude.max()  # field normalized to unit
    v_for_bessel = v * 0.581865

    # Calculate the inverse of J1 for each element of the normalized
    # amplitude 2D array
    F_g = apply_inverse_bessel_j1_2d(v_for_bessel)

    # Produce a hologram for SLM
    nx, ny = tuple(xy_grid_freq)
    numpxx, numpxy = tuple(xy_pixels)
    gx, gy = nx / (numpxx * 8e-3), ny / (numpxy * 8e-3)
    X, Y = np.meshgrid(xrange, yrange)

    hol = F_g * np.sin(phase + 2 * np.pi * (X * gx + Y * gy))
    hol = hol - hol.min()
    SLM = hol/hol.max()*255

    return SLM


def stack_matrix(twod_array: np.array, stack_configs: dict) -> np.array:

    stacked_array = twod_array.copy()
    if stack_configs.get('stack_direction') == "horizontal":
        if not stack_configs.get('stack_with'):
            black_array = np.zeros_like(twod_array)
            negative_array = -1 * twod_array
            stacked_array = [
                np.hstack((black_array, twod_array)),
                np.hstack((twod_array, negative_array))
            ]
        elif stack_configs.get('stack_with') == "negative":
            stacked_array = np.hstack((
                twod_array, -1 * twod_array
            ))
    else:
        if not stack_configs.get('stack_with'):
            black_array = np.zeros_like(twod_array)
            negative_array = -1 * twod_array
            stacked_array = [
                np.vstack((black_array, twod_array)),
                np.vstack((twod_array, negative_array))
            ]
        elif stack_configs.get('stack_with') == "negative":
            stacked_array = np.vstack((
                twod_array, -1 * twod_array
            ))

    return stacked_array


def main(light_beam_parameters: dict, hologram_parameters: dict, turbulence_parameters: dict) -> None:
    # set params to calculate complex field amplitudes
    beam_waist = light_beam_parameters.get('beam_waist')
    wave_number = k = 2 * np.pi / light_beam_parameters.get('wave_length')
    list_of_mode_indexes = light_beam_parameters.get('HO_mode_indexes')
    start_longitudinal_position = light_beam_parameters.get('longitudinal_pos')
    mode_short_name = light_beam_parameters.get('mode_name')
    divergence_switch = light_beam_parameters.get('is_divergent')

    px_array = hologram_parameters.get("XY_num_pixels")
    numxpx, numypx = px_array[0], px_array[1]

    # adjust x/y range according to stack params
    hol_stack_configs = hologram_parameters.get('stack_holograms')
    if hol_stack_configs.get('should_stack'):
        if hol_stack_configs.get('stack_direction') == "horizontal":
            numxpx = int(numxpx // 2)
        elif hol_stack_configs.get('stack_direction') == "vertical":
            numypx = int(numypx // 2)

    hrange = np.linspace(-1*(numxpx/2), (numxpx/2)-1, numxpx) * 8e-3
    vrange = np.linspace(-1*(numypx/2), (numypx/2)-1, numypx) * 8e-3

    # calculate transverse complex field amplitudes
    e_complex = [
        complex_field_amplitude(
            w0=beam_waist,
            k=wave_number,
            mode_indexes=curr_mode,
            z=start_longitudinal_position,
            xrange=hrange,
            yrange=vrange,
            mode_name=mode_short_name,
            divergence=divergence_switch
        ) for curr_mode in list_of_mode_indexes
    ]

    # now create holograms for each calculated amplitude
    if hologram_parameters.get('is_on'):
        freq_of_hologram_grid = hologram_parameters.get('XY_hol_grid_freq')
        slm_holograms = [
            calculate_hologram(
                e_complex=complex_field,
                xy_grid_freq=freq_of_hologram_grid,
                xy_pixels=[numxpx, numypx],
                xrange=hrange,
                yrange=vrange
            ) for complex_field in e_complex
        ]

        # stack if necessary
        if hol_stack_configs.get('should_stack'):
            slm_holograms = [
                stack_matrix(
                    twod_array=array,
                    stack_configs=hol_stack_configs
                ) for array in slm_holograms
            ]

        # save calculated holograms to .bitmap
        for slm, named_indexes in zip(slm_holograms, list_of_mode_indexes):
            curr_slm = slm.astype(np.uint8)
            image = Image.fromarray(curr_slm, 'L')
            now = datetime.now().isoformat(timespec='seconds')
            image.save(
                f"output/holograms/{mode_short_name}_{named_indexes[0]}{named_indexes[1]}_{now}.bmp"
            )

    # set params to produce turbulence screens
    npanels = turbulence_parameters.get('n_realizations')
    SR = turbulence_parameters.get('SR')
    freq_of_screen_grid = turbulence_parameters.get('XY_screen_grid_freq')
    turb_screen_px_size = turbulence_parameters.get('turb_pixel_size')
    turb_stack_configs = turbulence_parameters.get('stack_screens')
    px_array = turbulence_parameters.get("XY_num_pixels")

    # adjust x/y range according to stack params
    if turb_stack_configs.get('should_stack'):
        if turb_stack_configs.get('stack_direction') == "horizontal":
            px_array = halve_element(px_array, "horizontal")
        elif turb_stack_configs.get('stack_direction') == "vertical":
            px_array = halve_element(px_array, "vertical")

    if turbulence_parameters.get('is_on'):
        # generate screens
        for sr_val in SR:
            for panel in range(npanels):
                turb, ph_screen = generate_phase_screen(
                    SR=sr_val,
                    H=px_array[0][0],
                    V=px_array[0][1],
                    w0=beam_waist,
                    Pixel=turb_screen_px_size,
                    n=1,
                    nx=freq_of_screen_grid[0],
                    ny=freq_of_screen_grid[1],
                    seed=panel
                )

                if turb_stack_configs.get('should_stack'):
                    ph_screen = stack_matrix(twod_array=ph_screen, stack_configs=turb_stack_configs)

                # save phase screen
                if isinstance(ph_screen, list):
                    for ph, stype in zip(ph_screen, ['black', 'negative']):
                        ph_ = ph.astype(np.uint8)
                        image = Image.fromarray(ph_, 'L')
                        now = datetime.now().isoformat(timespec='minutes')
                        image.save(
                            f"output/turbulence_screens/SR={sr_val}_real={panel}_with={stype}_{now}.bmp"
                        )
                else:
                    ph_ = ph_screen.astype(np.uint8)
                    image = Image.fromarray(ph_, 'L')
                    now = datetime.now().isoformat(timespec='minutes')
                    image.save(
                        f"output/turbulence_screens/ASIS_RES={px_array[0][0]}X{px_array[0][1]}_SR={sr_val}_real={panel}_{now}.bmp"
                    )
                    if len(px_array)>1:
                        #ph_screen = -1*fourier_rescale(ph_screen, tuple(px_array[1]))
                        ph_screen = -1 * upscale_image_nearest(ph_screen, px_array[1][0], px_array[1][1])
                        ph_ = ph_screen.astype(np.uint8)
                        image = Image.fromarray(ph_, 'L')
                        image.save(
                            f"output/turbulence_screens/RESCALED_RES={px_array[1][0]}X{px_array[1][1]}_SR={sr_val}_real={panel}_{now}.bmp"
                        )


if __name__ == "__main__":
    # read parameters from config file
    with open("config.json") as config_file:
        config_items = json.load(config_file)

    beam_params = config_items.get("beam_params")
    hologram_params = config_items.get("hologram_params")
    turb_params = config_items.get("turb_params")

    main(beam_params, hologram_params, turb_params)