{
  description = "Traffic sign recognition project development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            python313Packages.pip
            python313Packages.virtualenv
          ];

          # Manylinux wheels (numpy, opencv-python, torch, ...) are compiled
          # against system shared libraries that a Nix shell doesn't expose
          # by default (there's no FHS /usr/lib). Point the loader at Nix's
          # own copies so those compiled extensions can find them at runtime.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
            pkgs.libxcb
            pkgs.libx11
            pkgs.libxext
            pkgs.libsm
            pkgs.libice
          ];

          # Nix provides the interpreter; pip-installed deps still live in a
          # project-local venv so `pip install` works as usual.
          shellHook = ''
            if [ ! -d .venv ]; then
              python -m venv .venv
            fi
            source .venv/bin/activate

            if [ requirements.txt -nt .venv/.deps-installed ]; then
              pip install -q -r requirements.txt
              touch .venv/.deps-installed
            fi
          '';
        };
      });
}
