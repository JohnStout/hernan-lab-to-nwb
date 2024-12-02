READ ME

TODO: 

        Make a base class that all decode_lab_code objects inherit. This will restrict organization across different coding patterns




image_utils: This is a group of functions for processing imaging data.
        It was designed for calcium imaging and is currently in progress.
        The most useful function is stacktiffs, which takes a bunch of .tif
        files and compiles them into a movie.
    
nwb_utils: This is currently a group of functions, but will be made into a class

    read_nwb: allows reading of nwb files

    write_nwb: a function to write nwb files

    generate_nwb_sheet: this is incomplete, but will take some  template and convert it into inputs for NWB files. This is extremely useful for times when you're running a lot of animals or recording a lot of sessions and most of the information is consistent between sessions. 

    nlx2nwb: take a neuralynx recorded dataset and convert it into an nwb file. 

    read_vt_data: will read video tracking data. Will probably be a group of separate functions because extraction methods may vary significantly between datasets.

