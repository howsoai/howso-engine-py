# Contributing

This Howso&trade; opensource project only accepts code contributions from individuals and organizations that have signed a contributor license agreement. For more information on contributing and for links to the individual and corporate CLAs, please visit: https://www.howso.com/cla

## Local Development and Testing

While developing, you may require the latest version of the howso-engine itself
installed and present in your environment. In addition, unit tests are designed
to use the embedded version of the howso-engine by default.

In source distributions of this package, the embedded howso-engine may not be
present. These artifacts can be obtained from the howso-client repository's
release artifacts and should be placed into the required location in your
development file-structure.

The file structure should look something like this:

    howso-engine-py/
        howso/
            client/
            direct/
            engine/
            ...
            howso-engine/    <- These files should be added here.
                migrations/
                    migrations.caml
                trainee/
                howso.caml
                version.json
            ...

To support this, by default, tests will ignore local configuration files
(`howso.yml`, et al) and the environment variables that may normally direct the
running `howso-engine` to a configuration (`HOWSO_CONFIG` and even
`XDG_CONFIG_HOME`).

This permits developers to have local configurations for normal operations of
the Howso product suite and still be able to test configuration-less operation.

### Testing against other configurations

If it is desirable to test other configurations as well. The environment
variable `TEST_OPTIONS` can include the option `USE_HOWSO_CONFIG`. Providing
this option restores the normal configuration-finding facilities to locate
a configuration that may already exist in your development environment.

#### Default test behavior

    > python -m pytest

This will test using only the embedded version of `howso-engine` and the
Amalgam shared libraries installed with `amalgam-lang`.

#### Normal, locally configured behavior

    > TEST_OPTIONS="USE_HOWSO_CONFIG" python -m pytest

This will be configured using the normal local configuration options which may
include a specific location to find the `howso-engine` and/or specific
versions of the Amalgam shared libraries.

If no such configuration is found, testing will fallback to the embedded
`howso-engine` and Amalgam shared libraries.
