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

          # Nix provides the interpreter; pip-installed deps still live in a
          # project-local venv so `pip install` works as usual.
          shellHook = ''
            if [ ! -d .venv ]; then
              python -m venv .venv
            fi
            source .venv/bin/activate
          '';
        };
      });
}
